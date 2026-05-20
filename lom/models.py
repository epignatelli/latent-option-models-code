"""Two composable primitives:

  LatentActionModel  — bidirectional encoder: (history, future[, condition]) → z via VQ
  DynamicsModel      — causal decoder:        (history, action[, goal])       → frame(s)

Compose to build LAM or LOM:

  LAM (baseline):
    lam          = LatentActionModel(codebook_size=n_actions, max_future_len=1)
    lam_dynamics = DynamicsModel(predict_sequence=False)

    z_act = lam(history, x_{t+1})
    x̂     = lam_dynamics(history, z_act)            → x_{t+1}

  LOM (proposed):
    option_lam   = LatentActionModel(codebook_size=num_options, max_future_len=k)
    action_lam   = LatentActionModel(codebook_size=n_actions,   max_future_len=1,
                                     condition_dim=latent_dim)
    lam_dynamics = DynamicsModel(predict_sequence=False)
    lom_dynamics = DynamicsModel(goal_dim=latent_dim, predict_sequence=False|True)

    z_opt = option_lam(history, sequence)                            # x_{t+1}…x_{t+k}
    z_act = action_lam(history, x_{t+1}, condition=z_opt)

    lam_dynamics(history, z_act)                                     → x̂_{t+1}   [LAM loss]
    lom_dynamics(history, z_act, goal=z_opt, n_steps=k)              → x̂_{t+k}   [LOM loss]
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .modules import (
    LayerNorm,
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
        max_context: int,
        latent_dim: int,
        codebook_size: int,
        max_future_len: int = 1,
        condition_dim: Optional[int] = None,
        vq_dropout: float = 0.1,
        vq_entropy_weight: float = 0.01,
        vq_beta: float = 0.25,
        vq_reset_thresh: int = 100,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        S = obs_h * obs_w
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.S = S
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_context = max_context
        self.has_condition = condition_dim is not None

        # temporal positions: history + optional cond token + OPT + future frames
        max_temporal_len = max_context + (1 if self.has_condition else 0) + 1 + max_future_len

        self.char_embed = nn.Embedding(vocab_size, d_model)
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
            history:   (B, c, H, W) long
            future:    (B, H, W) or (B, k, H, W) long
            condition: (B, condition_dim) optional
        Returns:
            z_q (B, latent_dim), vq loss dict, indices (B,)
        """
        B, c = history.shape[:2]

        if future.ndim == 3:
            future = future.unsqueeze(1)  # (B, 1, H, W)
        k = future.shape[1]

        hist_emb = self.char_embed(history.reshape(B, c, self.S))
        fut_emb = self.char_embed(future.reshape(B, k, self.S))
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

        opt_pos = c + (1 if condition is not None and self.has_condition else 0)
        opt_pooled = hidden[:, opt_pos, :, :].mean(dim=1)  # (B, D)
        z = self.ln_vq(self.vq_proj(opt_pooled))
        return self.vq(z)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# --- Dynamics Model -------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class DynamicsModel(SerialisableModule):
    """Causal STP-Transformer: (history, action[, goal]) → frame(s).

    action is broadcast-added to all input embeddings as the primary conditioning.
    goal is an optional secondary conditioning (e.g. z_option for hierarchical use).

    n_steps controls how many frames to predict.
    return_sequence controls whether all frames or only the last are returned.

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
        max_context: int,
        latent_dim: int,
        goal_dim: Optional[int] = None,
        predict_sequence: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        S = obs_h * obs_w
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.S = S
        self.d_model = d_model
        self.max_context = max_context
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        self.predict_sequence = predict_sequence

        self.char_embed = nn.Embedding(vocab_size, d_model)
        self.action_proj = nn.Linear(latent_dim, d_model, bias=bias)
        self.goal_proj = nn.Linear(goal_dim, d_model, bias=bias) if goal_dim is not None else None
        self.trunk = SpatioTemporalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_spatial_positions=S,
            max_temporal_len=max_context,
            dropout=dropout,
            bias=bias,
            causal_temporal=True,
        )
        self.ln_trunk = LayerNorm(d_model, bias)
        self.state_head = nn.Linear(d_model, vocab_size, bias=bias)

    def _cond(self, action: torch.Tensor, goal: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.action_proj(action)
        if goal is not None and self.goal_proj is not None:
            c = c + self.goal_proj(goal)
        return c.view(action.shape[0], 1, 1, self.d_model)  # (B, 1, 1, D) — broadcasts over (T, S)

    def forward(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        n_steps: int = 1,
        teacher_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history:        (B, c, H, W) long
            action:         (B, latent_dim)
            goal:           (B, goal_dim) optional — e.g. z_option for LOM dynamics
            n_steps:        number of frames to predict
            teacher_frames: (B, n_steps, H, W) long — teacher forcing, training only
        Returns:
            (B, S, vocab_size)            if predict_sequence=False
            (B, n_steps, S, vocab_size)   if predict_sequence=True
        """
        B, c = history.shape[:2]
        S = self.S
        cond = self._cond(action, goal)  # (B, 1, 1, D)

        if self.predict_sequence:
            if teacher_frames is not None:
                # Single forward pass with teacher forcing (training).
                # Input:  [h_0, …, h_{c-1}, f_0, …, f_{n-2}]
                # Targets at positions c-1 … c+n-2 predict f_0 … f_{n-1}
                inp = torch.cat([history, teacher_frames[:, :-1]], dim=1)  # (B, c+n-1, H, W)
                emb = self.char_embed(inp.reshape(B, c + n_steps - 1, S)) + cond
                hid = self.ln_trunk(self.trunk(emb))
                return self.state_head(hid[:, c - 1 : c + n_steps - 1, :, :])  # (B, n, S, V)
            else:
                # Autoregressive rollout (inference).
                frames, current = [], history
                for _ in range(n_steps):
                    emb = self.char_embed(current.reshape(B, current.shape[1], S)) + cond
                    hid = self.ln_trunk(self.trunk(emb))
                    logits = self.state_head(hid[:, -1, :, :])  # (B, S, V)
                    frames.append(logits)
                    next_f = logits.argmax(dim=-1).reshape(B, 1, self.obs_h, self.obs_w)
                    current = torch.cat([current[:, 1:], next_f], dim=1)  # slide window
                return torch.stack(frames, dim=1)  # (B, n, S, V)

        # Single-frame prediction.
        emb = self.char_embed(history.reshape(B, c, S)) + cond
        hid = self.ln_trunk(self.trunk(emb))
        return self.state_head(hid[:, -1, :, :])  # (B, S, V)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
