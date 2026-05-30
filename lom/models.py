"""Top-level LOM model classes.

Two concrete model variants, each self-contained:

  ReconstructionLOM — STT encoders + pixel-level reconstruction dynamics
  LatentLOM         — JEPA encoders + latent-space dynamics (EMA targets)

Both take (history, future) and return a dict of predictions and VQ info.
All building blocks (encoders, VQ, dynamics) live in lom.modules.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import LatentLOMCfg, ReconstructionLOMCfg  # noqa: F401  (re-exported for convenience)
from .modules import (
    DynamicsModel,
    EMAEncoder,
    JEPAEncoder,
    LatentActionModel,
    SerialisableModule,
    STTEncoder,
)


class ReconstructionLOM(SerialisableModule):
    """LOM with STT (bidirectional) encoders and pixel-level reconstruction dynamics.

    Architecture:
      option_lam   — STTEncoder: (history, future)           → z_opt via VQ
      action_lam   — STTEncoder: (history, future[:,0:1], z_opt) → z_act via VQ
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
        history: torch.Tensor,  # (B, c, H, W, 2)
        future:  torch.Tensor,  # (B, k, H, W, 2)  k=horizon; future[:,0:1] is next frame
    ) -> dict:
        next_frame = future[:, 0:1]
        z_opt, vq_opt, opt_idx = self.option_lam(history, future)
        z_act, vq_act, act_idx = self.action_lam(history, next_frame, condition=z_opt.detach())
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
      option_lam      — JEPAEncoder: (history, future)              → z_opt via LOM VQ
      action_lam      — JEPAEncoder: (history, future[:,0:1]) + z_opt → z_act via LAM VQ
      lam_dynamics    — causal transformer: (history, z_act)          → z_act_hat
      lom_dynamics    — causal transformer: (history, z_opt)          → z_opt_hat
      ema_action_enc  — EMA of action_lam.encoder.future_encoder      → z_act_target
      ema_option_enc  — EMA of option_lam.encoder.future_encoder      → z_opt_target

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
        history: torch.Tensor,  # (B, c, H, W, 2)
        future:  torch.Tensor,  # (B, k, H, W, 2)  k=horizon; future[:,0:1] is next frame
    ) -> dict:
        next_frame = future[:, 0:1]
        z_opt, vq_opt, opt_idx = self.option_lam(history, future)
        z_act, vq_act, act_idx = self.action_lam(history, next_frame, option_code=z_opt.detach())
        with torch.no_grad():
            z_act_target = self.ema_action_enc.encode(next_frame)
            z_opt_target = self.ema_option_enc.encode(future)
        z_act_hat = self.lam_dynamics(history, z_act)
        z_opt_hat = self.lom_dynamics(history, z_opt)
        return {
            "z_act_hat": z_act_hat, "z_act_target": z_act_target,
            "z_opt_hat": z_opt_hat, "z_opt_target": z_opt_target,
            "z_opt": z_opt, "z_act": z_act,
            "opt_idx": opt_idx, "act_idx": act_idx,
            "vq_opt": vq_opt, "vq_act": vq_act,
        }
