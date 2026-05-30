"""Top-level LOM model classes.

Two concrete model variants, each self-contained:

  ReconstructionLOM — STT encoders + pixel-level reconstruction dynamics
  LatentLOM         — JEPA encoders + latent-space dynamics (EMA targets)

Both take (history, future) and return a dict of predictions and VQ info.
All building blocks (encoders, VQ, dynamics) live in lom.modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modules import (
    DynamicsModel,
    EMAEncoder,
    JEPAEncoder,
    LayerNorm,
    SerialisableModule,
    STTEncoder,
    VectorQuantizer,
)


class ReconstructionLOM(SerialisableModule):
    """LOM with STT (bidirectional) encoders and pixel-level reconstruction dynamics.

    Architecture:
      opt_encoder  — STTEncoder: (history, future)                 → (B, d_model)
      opt_vq_proj  — Linear + LayerNorm → VectorQuantizer          → z_opt
      act_encoder  — STTEncoder: (history, next_frame, z_opt)      → (B, d_model)
      act_vq_proj  — Linear + LayerNorm → VectorQuantizer          → z_act
      lam_dynamics — causal transformer: (history, z_act)          → next_frame logits
      lom_dynamics — causal transformer: (history, z_act, z_opt)   → future logits

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
        enc_base = dict(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, patch_size=patch_size,
            dropout=dropout, bias=bias,
        )
        vq_kw = dict(
            dropout=vq_dropout, entropy_weight=vq_entropy_weight,
            vq_beta=vq_beta, vq_reset_thresh=vq_reset_thresh, ema_decay=vq_ema_decay,
        )
        dyn_base = dict(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, latent_dim=latent_dim,
            patch_size=patch_size, dropout=dropout, bias=bias,
        )

        # Option branch
        self.opt_encoder  = STTEncoder(**enc_base, horizon=horizon)
        self.opt_vq_proj  = nn.Linear(d_model, latent_dim, bias=bias)
        self.opt_ln_vq    = LayerNorm(latent_dim, bias)
        self.opt_vq       = VectorQuantizer(latent_dim=latent_dim, num_options=num_options, **vq_kw)

        # Action branch (condition = z_opt passed as token inside STTEncoder)
        self.act_encoder  = STTEncoder(**enc_base, horizon=1, condition_dim=latent_dim)
        self.act_vq_proj  = nn.Linear(d_model, latent_dim, bias=bias)
        self.act_ln_vq    = LayerNorm(latent_dim, bias)
        self.act_vq       = VectorQuantizer(latent_dim=latent_dim, num_options=n_actions, **vq_kw)

        # Dynamics
        self.lam_dynamics = DynamicsModel(**dyn_base, predict_sequence=False)
        self.lom_dynamics = DynamicsModel(**dyn_base, option_dim=latent_dim,
                                          predict_sequence=predict_sequence, horizon=horizon)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        history: torch.Tensor,  # (B, c, H, W, 2)
        future:  torch.Tensor,  # (B, k, H, W, 2)  k=horizon; future[:,0:1] is next frame
    ) -> dict:
        # Option code
        opt_pooled = self.opt_encoder(history, future)                     # (B, D)
        z_opt, vq_opt, opt_idx = self.opt_vq(self.opt_ln_vq(self.opt_vq_proj(opt_pooled)))

        # Action code (z_opt injected as condition token inside STTEncoder)
        next_frame = future[:, 0:1]
        act_pooled = self.act_encoder(history, next_frame, condition=z_opt.detach())  # (B, D)
        z_act, vq_act, act_idx = self.act_vq(self.act_ln_vq(self.act_vq_proj(act_pooled)))

        # Dynamics predictions
        lam_logits = self.lam_dynamics(history, z_act)
        if self.predict_sequence:
            lom_logits = self.lom_dynamics(history, z_act, option_code=z_opt,
                                           horizon=self.horizon, teacher_frames=future)
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
      opt_encoder  — JEPAEncoder: (history, future)                → (B, 2*d_model)
      opt_vq_proj  — Linear + LayerNorm → VectorQuantizer          → z_opt
      act_encoder  — JEPAEncoder: (history, next_frame)            → (B, 2*d_model)
      act_vq_proj  — Linear(2D+latent_dim) + LayerNorm → VQ       → z_act  [z_opt concat]
      lam_dynamics — causal transformer: (history, z_act)          → z_act_hat
      lom_dynamics — causal transformer: (history, z_opt)          → z_opt_hat
      ema_opt_enc  — EMA of opt_encoder                            → z_opt_target
      ema_act_enc  — EMA of act_encoder                            → z_act_target

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
        enc_base = dict(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, latent_dim=latent_dim,
            patch_size=patch_size, dropout=dropout, bias=bias,
        )
        vq_kw = dict(
            dropout=vq_dropout, entropy_weight=vq_entropy_weight,
            vq_beta=vq_beta, vq_reset_thresh=vq_reset_thresh, ema_decay=vq_ema_decay,
        )
        dyn_base = dict(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, latent_dim=latent_dim,
            patch_size=patch_size, dropout=dropout, bias=bias,
        )

        # Option branch
        self.opt_encoder  = JEPAEncoder(**enc_base, horizon=horizon)
        self.opt_vq_proj  = nn.Linear(self.opt_encoder.out_dim, latent_dim, bias=bias)
        self.opt_ln_vq    = LayerNorm(latent_dim, bias)
        self.opt_vq       = VectorQuantizer(latent_dim=latent_dim, num_options=num_options, **vq_kw)

        # Action branch (z_opt concatenated before projection)
        self.act_encoder  = JEPAEncoder(**enc_base, horizon=1)
        self.act_vq_proj  = nn.Linear(self.act_encoder.out_dim + latent_dim, latent_dim, bias=bias)
        self.act_ln_vq    = LayerNorm(latent_dim, bias)
        self.act_vq       = VectorQuantizer(latent_dim=latent_dim, num_options=n_actions, **vq_kw)

        # Dynamics
        self.lam_dynamics = DynamicsModel(**dyn_base, predict_sequence=False,
                                          predict_latent=True, target_dim=latent_dim)
        self.lom_dynamics = DynamicsModel(**dyn_base, predict_sequence=False,
                                          predict_latent=True, target_dim=latent_dim)

        # EMA target encoders
        self.ema_opt_enc  = EMAEncoder(self.opt_encoder, decay=ema_decay)
        self.ema_act_enc  = EMAEncoder(self.act_encoder, decay=ema_decay)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def update_ema(self) -> None:
        self.ema_opt_enc.update(self.opt_encoder)
        self.ema_act_enc.update(self.act_encoder)

    def forward(
        self,
        history: torch.Tensor,  # (B, c, H, W, 2)
        future:  torch.Tensor,  # (B, k, H, W, 2)  k=horizon; future[:,0:1] is next frame
    ) -> dict:
        # Option code
        opt_pooled = self.opt_encoder(history, future)                     # (B, 2D)
        z_opt, vq_opt, opt_idx = self.opt_vq(self.opt_ln_vq(self.opt_vq_proj(opt_pooled)))

        # Action code (z_opt concatenated at projection input)
        next_frame = future[:, 0:1]
        act_pooled = self.act_encoder(history, next_frame)                 # (B, 2D)
        act_in = torch.cat([act_pooled, z_opt.detach()], dim=-1)           # (B, 2D + latent_dim)
        z_act, vq_act, act_idx = self.act_vq(self.act_ln_vq(self.act_vq_proj(act_in)))

        # EMA targets (no grad)
        with torch.no_grad():
            z_act_target = self.ema_act_enc.encode(next_frame)
            z_opt_target = self.ema_opt_enc.encode(future)

        # Dynamics predictions
        z_act_hat = self.lam_dynamics(history, z_act)
        z_opt_hat = self.lom_dynamics(history, z_opt)

        return {
            "z_act_hat": z_act_hat, "z_act_target": z_act_target,
            "z_opt_hat": z_opt_hat, "z_opt_target": z_opt_target,
            "z_opt": z_opt, "z_act": z_act,
            "opt_idx": opt_idx, "act_idx": act_idx,
            "vq_opt": vq_opt, "vq_act": vq_act,
        }
