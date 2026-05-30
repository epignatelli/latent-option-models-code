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

from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .modules import (
    BlockMask,
    LayerNorm,
    PatchEmbedding,
    ScreenTokeniser,
    SpatioTemporalTransformer,
    VectorQuantizer,
    SerialisableModule,
    _get_opt_block_mask,
)
from .config import ReconstructionLOMCfg, LatentLOMCfg

# --------------------------------------------------------------------------- #
# --- Models ---------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class ReconstructionLOM(SerialisableModule):
    """LOM with STT (bidirectional) encoders and pixel-level reconstruction dynamics.

    Architecture:
      option_lam   — STTEncoder: (history, sequence)           → z_opt via VQ
      action_lam   — STTEncoder: (history, next_frame, z_opt)  → z_act via VQ
      lam_dynamics — causal transformer: (history, z_act)      → next_frame logits
      lom_dynamics — causal transformer: (history, z_act, z_opt) → future logits

    forward() returns raw predictions and VQ info; loss computation is external.
    """

    def __init__(
        self,
        vocab_size: int,
        obs_h: int,
        obs_w: int,
        n_actions: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        context_length: int,
        horizon: int,
        latent_dim: int,
        num_options: int,
        patch_size: int = 1,
        predict_sequence: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
        vq_dropout: float = 0.1,
        vq_entropy_weight: float = 0.01,
        vq_beta: float = 0.25,
        vq_reset_thresh: int = 100,
        vq_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.horizon = horizon
        self.predict_sequence = predict_sequence
        base = dict(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, latent_dim=latent_dim,
            patch_size=patch_size, dropout=dropout, bias=bias,
        )
        vq = dict(
            vq_dropout=vq_dropout, vq_entropy_weight=vq_entropy_weight,
            vq_beta=vq_beta, vq_reset_thresh=vq_reset_thresh, vq_ema_decay=vq_ema_decay,
        )
        self.option_lam = LatentActionModel(**base, codebook_size=num_options, horizon=horizon, **vq)
        self.action_lam = LatentActionModel(**base, codebook_size=n_actions, horizon=1,
                                            condition_dim=latent_dim, **vq)
        self.lam_dynamics = DynamicsModel(**base, predict_sequence=False)
        self.lom_dynamics = DynamicsModel(**base, option_dim=latent_dim,
                                          predict_sequence=predict_sequence, horizon=horizon)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        history:    torch.Tensor,  # (B, c, H, W, 2)
        next_frame: torch.Tensor,  # (B, 1, H, W, 2)
        future:     torch.Tensor,  # (B, 1, H, W, 2)  target for lom_dynamics
        sequence:   torch.Tensor,  # (B, k, H, W, 2)  full horizon for option_lam
    ) -> dict:
        z_opt, vq_opt, opt_idx = self.option_lam(history, sequence)
        z_act, vq_act, act_idx = self.action_lam(history, next_frame, condition=z_opt.detach())
        lam_logits = self.lam_dynamics(history, z_act)
        if self.predict_sequence:
            lom_logits = self.lom_dynamics(history, z_act, option_code=z_opt,
                                           horizon=self.horizon, teacher_frames=sequence)
        else:
            lom_logits = self.lom_dynamics(history, z_act, option_code=z_opt, horizon=1)
        return {
            "lam_logits": lam_logits,
            "lom_logits": lom_logits,
            "z_opt": z_opt, "z_act": z_act,
            "opt_idx": opt_idx, "act_idx": act_idx,
            "vq_opt": vq_opt, "vq_act": vq_act,
        }


class LatentLOM(SerialisableModule):
    """LOM with JEPA encoders (separate causal past/future) and latent-space dynamics.

    Architecture:
      option_lam      — JEPAEncoder: (history, sequence)           → z_opt via LOM VQ
      action_lam      — JEPAEncoder: (history, next_frame) + z_opt → z_act via LAM VQ
      lam_dynamics    — causal transformer: (history, z_act)        → z_act_hat
      lom_dynamics    — causal transformer: (history, z_opt)        → z_opt_hat
      ema_action_enc  — EMA of action_lam.future_encoder            → z_act_target
      ema_option_enc  — EMA of option_lam.future_encoder            → z_opt_target

    forward() returns predictions and EMA targets; losses are external.
    Call update_ema() after each optimiser step.
    """

    def __init__(
        self,
        vocab_size: int,
        obs_h: int,
        obs_w: int,
        n_actions: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        context_length: int,
        horizon: int,
        latent_dim: int,
        num_options: int,
        patch_size: int = 1,
        ema_decay: float = 0.996,
        dropout: float = 0.0,
        bias: bool = False,
        vq_dropout: float = 0.1,
        vq_entropy_weight: float = 0.01,
        vq_beta: float = 0.25,
        vq_reset_thresh: int = 100,
        vq_ema_decay: float = 0.99,
    ):
        super().__init__()
        base = dict(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, latent_dim=latent_dim,
            patch_size=patch_size, dropout=dropout, bias=bias,
        )
        vq = dict(
            vq_dropout=vq_dropout, vq_entropy_weight=vq_entropy_weight,
            vq_beta=vq_beta, vq_reset_thresh=vq_reset_thresh, vq_ema_decay=vq_ema_decay,
        )
        self.option_lam = LatentActionModel(**base, codebook_size=num_options, horizon=horizon,
                                            two_encoder=True, **vq)
        self.action_lam = LatentActionModel(**base, codebook_size=n_actions, horizon=1,
                                            option_code_dim=latent_dim, two_encoder=True, **vq)
        self.lam_dynamics = DynamicsModel(**base, predict_sequence=False,
                                          predict_latent=True, target_dim=latent_dim)
        self.lom_dynamics = DynamicsModel(**base, predict_sequence=False,
                                          predict_latent=True, target_dim=latent_dim)
        self.ema_option_enc = EMAEncoder(self.option_lam.encoder, decay=ema_decay)  # type: ignore[arg-type]
        self.ema_action_enc = EMAEncoder(self.action_lam.encoder, decay=ema_decay)  # type: ignore[arg-type]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def update_ema(self) -> None:
        self.ema_option_enc.update(self.option_lam.encoder)  # type: ignore[arg-type]
        self.ema_action_enc.update(self.action_lam.encoder)  # type: ignore[arg-type]

    def forward(
        self,
        history:    torch.Tensor,  # (B, c, H, W, 2)
        next_frame: torch.Tensor,  # (B, 1, H, W, 2)
        sequence:   torch.Tensor,  # (B, k, H, W, 2)  full horizon for option_lam
    ) -> dict:
        z_opt, vq_opt, opt_idx = self.option_lam(history, sequence)
        z_act, vq_act, act_idx = self.action_lam(history, next_frame, option_code=z_opt.detach())
        with torch.no_grad():
            z_act_target = self.ema_action_enc.encode(next_frame)
            z_opt_target = self.ema_option_enc.encode(sequence)
        z_act_hat = self.lam_dynamics(history, z_act)
        z_opt_hat = self.lom_dynamics(history, z_opt)
        return {
            "z_act_hat": z_act_hat, "z_act_target": z_act_target,
            "z_opt_hat": z_opt_hat, "z_opt_target": z_opt_target,
            "z_opt": z_opt, "z_act": z_act,
            "opt_idx": opt_idx, "act_idx": act_idx,
            "vq_opt": vq_opt, "vq_act": vq_act,
        }




# --------------------------------------------------------------------------- #
# --- Encoders -------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class STTEncoder(nn.Module):
    """Single-pass bidirectional encoder: (history, future[, condition]) → (B, d_model).

    Concatenates [history, (condition,) OPT, future] and runs a bidirectional
    SpatioTemporalTransformer. OPT is masked from attending to future frames;
    its spatial mean is the pooled representation.
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
        horizon: int = 1,
        patch_size: int = 1,
        condition_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.horizon = horizon
        self.has_condition = condition_dim is not None

        self.tokeniser = ScreenTokeniser()
        self.embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        S = self.S = self.embed.n_tokens

        extra = 1 if self.has_condition else 0
        self.cond_proj = (
            nn.Linear(condition_dim, d_model, bias=bias) if self.has_condition else None
        )
        self.opt_token = nn.Parameter(torch.randn(1, 1, S, d_model) * 0.02)
        self.transformer = SpatioTemporalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_spatial_positions=S,
            max_temporal_len=context_length + extra + 1 + horizon,
            dropout=dropout,
            bias=bias,
            causal_temporal=False,
        )

    @property
    def out_dim(self) -> int:
        return self.d_model

    def _build_block_mask(self, c: int, k: int, device: torch.device) -> BlockMask | None:
        if device.type != "cuda":
            return None  # CPU path (tests): skip OPT mask, flex_attention unavailable
        extra = 1 if self.has_condition else 0
        T = c + extra + 1 + k
        opt_pos = c + extra
        return _get_opt_block_mask(T, opt_pos, device)

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history:   (B, c, H, W, 2) uint8
            future:    (B, H, W, 2) or (B, k, H, W, 2) uint8
            condition: (B, condition_dim) optional
        Returns:
            (B, d_model)
        """
        B, c = history.shape[:2]
        history = self.tokeniser(history)
        future = self.tokeniser(future)
        if future.ndim == 3:
            future = future.unsqueeze(1)
        k = future.shape[1]

        hist_emb = self.embed(history)  # (B, c, S, D)
        fut_emb = self.embed(future)    # (B, k, S, D)
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
        hidden = self.transformer(seq, temporal_mask=self._build_block_mask(c, k, seq.device))

        opt_pos = c + (1 if self.has_condition else 0)
        return hidden[:, opt_pos, :, :].mean(dim=1)  # (B, D)


class JEPAEncoder(nn.Module):
    """JEPA encoder: past and future encoded by separate causal transformers.

    past_encoder:   history  → last-frame spatial mean → (B, D)
    future_encoder: future   → last-frame spatial mean → (B, D)
    Returns concat of both → (B, 2 * d_model).

    No shared weights between past and future; no cross-attention.
    Condition (z_opt for action_lam) is injected at the VQ level in
    LatentActionModel, not here.

    encode(frames) runs future_encoder only — used by EMAEncoder for targets.
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
        horizon: int = 1,
        patch_size: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
        **_kwargs,  # absorb unused kwargs (e.g. condition_dim from STT path)
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.tokeniser = ScreenTokeniser()
        self.embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        S = self.S = self.embed.n_tokens

        self.past_encoder = SpatioTemporalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=S, max_temporal_len=context_length,
            dropout=dropout, bias=bias, causal_temporal=True,
        )
        self.future_encoder = SpatioTemporalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=S, max_temporal_len=horizon,
            dropout=dropout, bias=bias, causal_temporal=True,
        )
        self.proj_target = nn.Linear(d_model, latent_dim, bias=bias)
        self.ln_target = LayerNorm(latent_dim, bias)

    @property
    def out_dim(self) -> int:
        return 2 * self.d_model

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames with future_encoder → last-frame spatial mean → proj.

        Used by EMAEncoder to produce JEPA targets.
        frames: (B, H, W, 2) or (B, k, H, W, 2) uint8
        """
        tokens = self.tokeniser(frames)
        if tokens.ndim == 3:
            tokens = tokens.unsqueeze(1)
        emb = self.embed(tokens)                                    # (B, k, S, D)
        out = self.future_encoder(emb)                              # (B, k, S, D)
        pooled = out[:, -1, :, :].mean(dim=1)                      # (B, D) last frame
        return self.ln_target(self.proj_target(pooled))             # (B, latent_dim)

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        condition: Optional[torch.Tensor] = None,  # unused; kept for API compat
    ) -> torch.Tensor:
        """
        Args:
            history: (B, c, H, W, 2) uint8
            future:  (B, H, W, 2) or (B, k, H, W, 2) uint8
        Returns:
            (B, 2 * d_model)
        """
        history = self.tokeniser(history)
        future  = self.tokeniser(future)
        if future.ndim == 3:
            future = future.unsqueeze(1)

        hist_emb = self.embed(history)                              # (B, c, S, D)
        ctx_pooled = self.past_encoder(hist_emb)[:, -1, :, :].mean(dim=1)   # (B, D)

        fut_emb  = self.embed(future)                               # (B, k, S, D)
        tgt_pooled = self.future_encoder(fut_emb)[:, -1, :, :].mean(dim=1)  # (B, D)

        return torch.cat([ctx_pooled, tgt_pooled], dim=-1)          # (B, 2D)


class EMAEncoder(nn.Module):
    """EMA shadow of a JEPAEncoder used as the JEPA target encoder.

    Parameters are not trained by the optimizer; updated each step via exponential
    moving average of the online JEPAEncoder.
    """

    def __init__(self, base: JEPAEncoder, decay: float = 0.996):
        super().__init__()
        self.encoder = deepcopy(base)
        self.decay = decay
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: JEPAEncoder) -> None:
        for e, o in zip(self.encoder.parameters(), online.parameters()):
            e.data.mul_(self.decay).add_(o.data, alpha=1 - self.decay)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(frames)


# --------------------------------------------------------------------------- #
# --- Latent Action Model --------------------------------------------------- #
# --------------------------------------------------------------------------- #


class LatentActionModel(SerialisableModule):
    """Frame encoder + VQ: (history, future[, condition]) → z via VQ.

    Encoder is selected by two_encoder:
      False (default): STTEncoder   — single pass with masked OPT token
      True:            JEPAEncoder  — context/target encoded separately, concat-pooled
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
        condition_dim: Optional[int] = None,  # STT only: condition token in encoder
        option_code_dim: Optional[int] = None,  # JEPA only: z_opt concatenated at VQ input
        two_encoder: bool = False,
        vq_dropout: float = 0.1,
        vq_entropy_weight: float = 0.01,
        vq_beta: float = 0.25,
        vq_reset_thresh: int = 100,
        vq_ema_decay: float = 0.99,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.context_length = context_length
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.horizon = horizon
        self.patch_size = patch_size
        self.condition_dim = condition_dim
        self.option_code_dim = option_code_dim
        self.two_encoder = two_encoder
        self.vq_dropout = vq_dropout
        self.vq_entropy_weight = vq_entropy_weight
        self.vq_beta = vq_beta
        self.vq_reset_thresh = vq_reset_thresh
        self.vq_ema_decay = vq_ema_decay
        self.dropout = dropout
        self.bias = bias

        encoder_cls = JEPAEncoder if two_encoder else STTEncoder
        encoder_kwargs: dict = dict(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, horizon=horizon,
            patch_size=patch_size, dropout=dropout, bias=bias,
        )
        if two_encoder:
            encoder_kwargs["latent_dim"] = latent_dim
        else:
            encoder_kwargs["condition_dim"] = condition_dim  # STT handles condition in encoder
        self.encoder = encoder_cls(**encoder_kwargs)
        vq_in_dim = self.encoder.out_dim + (option_code_dim or 0)
        self.vq_proj = nn.Linear(vq_in_dim, latent_dim, bias=bias)
        self.ln_vq = LayerNorm(latent_dim, bias)
        self.vq = VectorQuantizer(
            latent_dim=latent_dim,
            num_options=codebook_size,
            dropout=vq_dropout,
            entropy_weight=vq_entropy_weight,
            vq_beta=vq_beta,
            vq_reset_thresh=vq_reset_thresh,
            ema_decay=vq_ema_decay,
        )

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        option_code: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Args:
            history:     (B, c, H, W, 2) uint8/long
            future:      (B, H, W, 2) or (B, k, H, W, 2)
            condition:   (B, condition_dim) — STT encoder conditioning (token-based)
            option_code: (B, latent_dim)   — JEPA VQ-level conditioning (concat before proj)
        Returns:
            z_q (B, latent_dim), vq loss dict, indices (B,)
        """
        pooled = self.encoder(history, future, condition)
        if option_code is not None:
            pooled = torch.cat([pooled, option_code], dim=-1)
        z = self.ln_vq(self.vq_proj(pooled))
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
        predict_latent: bool = False,
        target_dim: Optional[int] = None,
        horizon: int = 1,
        patch_size: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert not (predict_latent and target_dim is None), \
            "target_dim is required when predict_latent=True"
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
        self.predict_latent = predict_latent
        self.target_dim = target_dim
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
        if predict_latent:
            # Latent prediction head: mean pool over spatial tokens → target latent
            self.latent_head = nn.Linear(d_model, target_dim, bias=bias)
            self.ln_latent = LayerNorm(target_dim, bias)
        else:
            # Observable prediction head: each patch token predicts patch_size² characters
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
            STT mode (predict_latent=False):
              (B, H*W, vocab_size)            if predict_sequence=False
              (B, horizon, H*W, vocab_size)   if predict_sequence=True
            JEPA mode (predict_latent=True):
              (B, target_dim)                 if predict_sequence=False
              (B, horizon, target_dim)        if predict_sequence=True
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
                hid = self.ln_trunk(self.trunk(emb))           # (B, c+n-1, S, D)
                hid_seq = hid[:, c - 1 : c + horizon - 1]     # (B, n, S, D)
                if self.predict_latent:
                    return self.ln_latent(self.latent_head(hid_seq.mean(dim=2)))  # (B, n, target_dim)
                return self._unpatch_logits(self.state_head(hid_seq))              # (B, n, H*W, V)
            else:
                frames, current = [], history
                for _ in range(horizon):
                    emb = self.embed(current) + cond
                    hid = self.ln_trunk(self.trunk(emb))
                    if self.predict_latent:
                        frames.append(self.ln_latent(self.latent_head(hid[:, -1].mean(dim=1))))  # (B, target_dim)
                    else:
                        logits = self._unpatch_logits(self.state_head(hid[:, -1]))  # (B, H*W, V)
                        frames.append(logits)
                        next_f = logits.argmax(dim=-1).reshape(B, 1, self.obs_h, self.obs_w)
                        current = torch.cat([current[:, 1:], next_f], dim=1)
                return torch.stack(frames, dim=1)  # (B, n, H*W, V) or (B, n, target_dim)

        emb = self.embed(history) + cond
        hid = self.ln_trunk(self.trunk(emb))
        if self.predict_latent:
            return self.ln_latent(self.latent_head(hid[:, -1].mean(dim=1)))  # (B, target_dim)
        return self._unpatch_logits(self.state_head(hid[:, -1]))              # (B, H*W, V)
