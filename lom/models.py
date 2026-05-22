"""Two composable primitives:

  LatentActionModel  — bidirectional encoder: (history, future[, condition]) → z via VQ
  DynamicsModel      — causal decoder:        (history, action[, option_code]) → frame(s)

Compose to build LAM or LOM:

  LAM (baseline):
    lam          = LatentActionModel(codebook_size=n_actions, horizon=1)
    lam_dynamics = DynamicsModel(predict_sequence=False)

    z_act = lam(history, x_{t+1})
    x̂     = lam_dynamics(history, z_act)            → x_{t+1}

  LOM (proposed):
    option_lam   = LatentActionModel(codebook_size=num_options, horizon=k)
    action_lam   = LatentActionModel(codebook_size=n_actions,   horizon=1,
                                     condition_dim=latent_dim)
    lam_dynamics = DynamicsModel(predict_sequence=False)
    lom_dynamics = DynamicsModel(option_dim=latent_dim, predict_sequence=False|True,
                                 horizon=k)

    z_opt = option_lam(history, sequence)                            # x_{t+1}…x_{t+k}
    z_act = action_lam(history, x_{t+1}, condition=z_opt)

    lam_dynamics(history, z_act)                                        → x̂_{t+1}   [LAM loss]
    lom_dynamics(history, z_act, option_code=z_opt, horizon=k)         → x̂_{t+k}   [LOM loss]
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .modules import (
    LayerNorm,
    PatchEmbedding,
    ScreenTokeniser,
    SpatioTemporalTransformer,
    VectorQuantizer,
    SerialisableModule,
)

# --------------------------------------------------------------------------- #
# --- Latent Action Model --------------------------------------------------- #
# --------------------------------------------------------------------------- #


class LatentActionModel(SerialisableModule):
    """Bidirectional STP-Transformer: (history, future[, condition]) → z via VQ.

    future can be a single frame (B, H, W) or a sequence (B, k, H, W).
    When future is a sequence the encoder sees the full trajectory, not just the endpoint.

    condition is an optional latent vector — e.g. z_option when training an action LAM
    on top of a frozen option code.

    Sequence layout without condition:
        [h_0, …, h_{c-1},  OPT,  f_0, …, f_{k-1}]

    Sequence layout with condition:
        [h_0, …, h_{c-1},  z_cond,  OPT,  f_0, …, f_{k-1}]

    OPT is masked from attending to all future frames, forcing it to encode the
    transition from history to future rather than copying future content.
    """

    def __init__(
        self,
        vocab_size: int,
        obs_h: int,
        obs_w: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        context_length: int,
        latent_dim: int,
        codebook_size: int,
        horizon: int = 1,
        patch_size: int = 1,
        condition_dim: Optional[int] = None,
        vq_dropout: float = 0.1,
        vq_entropy_weight: float = 0.01,
        vq_beta: float = 0.25,
        vq_reset_thresh: int = 100,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.context_length = context_length
        self.codebook_size = codebook_size
        self.horizon = horizon
        self.patch_size = patch_size
        self.condition_dim = condition_dim
        self.vq_dropout = vq_dropout
        self.vq_entropy_weight = vq_entropy_weight
        self.vq_beta = vq_beta
        self.vq_reset_thresh = vq_reset_thresh
        self.dropout = dropout
        self.bias = bias
        self.has_condition = condition_dim is not None

        self.tokeniser = ScreenTokeniser()
        self.embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        S = self.S = self.embed.n_tokens

        # temporal positions: history + optional cond token + OPT + future frames
        max_temporal_len = context_length + (1 if self.has_condition else 0) + 1 + horizon

        self.cond_proj = (
            nn.Linear(condition_dim, d_model, bias=bias) if self.has_condition else None
        )
        self.opt_token = nn.Parameter(torch.randn(1, 1, S, d_model) * 0.02)
        self.transformer = SpatioTemporalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_spatial_positions=S,
            max_temporal_len=max_temporal_len,
            dropout=dropout,
            bias=bias,
            causal_temporal=False,
        )
        self.vq_proj = nn.Linear(d_model, latent_dim, bias=bias)
        self.ln_vq = LayerNorm(latent_dim, bias)
        self.vq = VectorQuantizer(
            latent_dim=latent_dim,
            num_options=codebook_size,
            dropout=vq_dropout,
            entropy_weight=vq_entropy_weight,
            vq_beta=vq_beta,
            vq_reset_thresh=vq_reset_thresh,
        )

    def _build_mask(self, c: int, k: int, device: torch.device) -> torch.Tensor:
        extra = 1 if self.has_condition else 0
        T = c + extra + 1 + k
        opt_pos = c + extra
        mask = torch.zeros(T, T, device=device)
        mask[opt_pos, opt_pos + 1 :] = float("-inf")  # OPT cannot attend to any future frame
        return mask.view(1, 1, T, T)

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Args:
            history:   (B, c, H, W, 2) uint8/long — stacked (char, color)
            future:    (B, H, W, 2) or (B, k, H, W, 2)
            condition: (B, condition_dim) optional
        Returns:
            z_q (B, latent_dim), vq loss dict, indices (B,)
        """
        B, c = history.shape[:2]

        history = self.tokeniser(history)   # (B, c, H, W) long
        future  = self.tokeniser(future)    # (B, H, W) or (B, k, H, W) long
        if future.ndim == 3:
            future = future.unsqueeze(1)    # (B, 1, H, W)
        k = future.shape[1]

        hist_emb = self.embed(history)           # (B, c, S, D)
        fut_emb  = self.embed(future)            # (B, k, S, D)
        opt_emb = self.opt_token.expand(B, 1, self.S, self.d_model)

        parts = [hist_emb]
        if condition is not None and self.cond_proj is not None:
            cond_tok = (
                self.cond_proj(condition)
                .view(B, 1, 1, self.d_model)
                .expand(B, 1, self.S, self.d_model)
            )
            parts.append(cond_tok)
        parts += [opt_emb, fut_emb]

        seq = torch.cat(parts, dim=1)
        hidden = self.transformer(seq, temporal_mask=self._build_mask(c, k, history.device))

        opt_pos = c + (1 if self.has_condition else 0)
        opt_pooled = hidden[:, opt_pos, :, :].mean(dim=1)  # (B, D)
        z = self.ln_vq(self.vq_proj(opt_pooled))
        return self.vq(z)



# --------------------------------------------------------------------------- #
# --- Dynamics Model -------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class DynamicsModel(SerialisableModule):
    """Causal STP-Transformer: (history, action[, option_code]) → frame(s).

    action is broadcast-added to all input embeddings as the primary conditioning.
    option_code is an optional secondary conditioning (z_option for LOM dynamics).

    horizon controls how many frames to predict.

    Training: pass teacher_frames for efficient teacher-forced sequence prediction.
    Inference: leave teacher_frames=None; single-step is a plain forward pass,
               multi-step falls back to autoregressive rollout.
    """

    def __init__(
        self,
        vocab_size: int,
        obs_h: int,
        obs_w: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        context_length: int,
        latent_dim: int,
        option_dim: Optional[int] = None,
        predict_sequence: bool = False,
        horizon: int = 1,
        patch_size: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.context_length = context_length
        self.latent_dim = latent_dim
        self.option_dim = option_dim
        self.predict_sequence = predict_sequence
        self.horizon = horizon
        self.patch_size = patch_size
        self.dropout = dropout
        self.bias = bias

        self.tokeniser = ScreenTokeniser()
        self.embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        S = self.S = self.embed.n_tokens

        max_temporal_len = context_length + horizon - 1 if predict_sequence else context_length

        self.action_proj = nn.Linear(latent_dim, d_model, bias=bias)
        self.goal_proj = (
            nn.Linear(option_dim, d_model, bias=bias) if option_dim is not None else None
        )
        self.trunk = SpatioTemporalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_spatial_positions=S,
            max_temporal_len=max_temporal_len,
            dropout=dropout,
            bias=bias,
            causal_temporal=True,
        )
        self.ln_trunk = LayerNorm(d_model, bias)
        # each patch token predicts all patch_size² characters in its patch
        self.state_head = nn.Linear(d_model, vocab_size * patch_size ** 2, bias=bias)

    def _cond(self, action: torch.Tensor, option_code: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.action_proj(action)
        if option_code is not None and self.goal_proj is not None:
            c = c + self.goal_proj(option_code)
        return c.view(action.shape[0], 1, 1, self.d_model)  # (B, 1, 1, D) — broadcasts over (T, S)

    def _unpatch_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert patch-level logits to character-level logits.

        Args:
            logits: (..., n_tokens, patch_size² * vocab_size)
        Returns:
            (..., H*W, vocab_size)
        """
        P = self.patch_size
        if P == 1:
            return logits
        H, W, V = self.obs_h, self.obs_w, self.vocab_size
        prefix = logits.shape[:-2]
        # (..., n_tokens, P²*V) → (N, H/P, W/P, P, P, V) → (N, H, W, V) → (..., H*W, V)
        flat = logits.reshape(-1, H // P, W // P, P, P, V)
        flat = flat.permute(0, 1, 3, 2, 4, 5).contiguous()   # (N, H/P, P, W/P, P, V)
        flat = flat.reshape(-1, H * W, V)
        return flat.reshape(*prefix, H * W, V)

    def forward(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        option_code: Optional[torch.Tensor] = None,
        horizon: int = 1,
        teacher_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history:        (B, c, H, W, 2) uint8/long — stacked (char, color)
            action:         (B, latent_dim)
            option_code:    (B, goal_dim) optional — z_option for LOM dynamics
            horizon:        number of frames to predict
            teacher_frames: (B, horizon, H, W, 2) — teacher forcing, training only
        Returns:
            (B, H*W, vocab_size)            if predict_sequence=False
            (B, horizon, H*W, vocab_size)   if predict_sequence=True
        """
        B, c = history.shape[:2]
        history = self.tokeniser(history)  # (B, c, H, W) long
        if teacher_frames is not None:
            teacher_frames = self.tokeniser(teacher_frames)  # (B, horizon, H, W) long
        cond = self._cond(action, option_code)  # (B, 1, 1, D)

        if self.predict_sequence:
            if teacher_frames is not None:
                inp = torch.cat([history, teacher_frames[:, :-1]], dim=1)  # (B, c+n-1, H, W)
                emb = self.embed(inp) + cond
                hid = self.ln_trunk(self.trunk(emb))
                logits = self.state_head(hid[:, c - 1 : c + horizon - 1])  # (B, n, S, P²V)
                return self._unpatch_logits(logits)                          # (B, n, H*W, V)
            else:
                frames, current = [], history
                for _ in range(horizon):
                    emb = self.embed(current) + cond
                    hid = self.ln_trunk(self.trunk(emb))
                    logits = self._unpatch_logits(self.state_head(hid[:, -1]))  # (B, H*W, V)
                    frames.append(logits)
                    next_f = logits.argmax(dim=-1).reshape(B, 1, self.obs_h, self.obs_w)
                    current = torch.cat([current[:, 1:], next_f], dim=1)
                return torch.stack(frames, dim=1)  # (B, n, H*W, V)

        emb = self.embed(history) + cond
        hid = self.ln_trunk(self.trunk(emb))
        return self._unpatch_logits(self.state_head(hid[:, -1]))  # (B, H*W, V)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
