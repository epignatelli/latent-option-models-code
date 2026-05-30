"""Latent Action Model, Observable Transition Model, and Latent Transition Model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .modules import LayerNorm, PatchEmbedding, SpatioTemporalTransformer, VectorQuantizer
from .modules import SerialisableModule
from .tokeniser import ScreenTokeniser


class LatentActionModel(SerialisableModule):
    """VQ bottleneck: encoded representation → discrete latent code.

    Linear(in_dim → latent_dim) + LayerNorm + VectorQuantizer.
    Takes a pre-computed encoder pooling and returns (z_q, vq_dict, indices).
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        num_options: int,
        bias: bool = False,
        vq_dropout: float = 0.1,
        vq_entropy_weight: float = 0.01,
        vq_beta: float = 0.25,
        vq_reset_thresh: int = 100,
        vq_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.lam = nn.Sequential(
            nn.Linear(in_dim, latent_dim, bias=bias),
            LayerNorm(latent_dim, bias),
            VectorQuantizer(
                latent_dim=latent_dim,
                num_options=num_options,
                dropout=vq_dropout,
                entropy_weight=vq_entropy_weight,
                vq_beta=vq_beta,
                vq_reset_thresh=vq_reset_thresh,
                ema_decay=vq_ema_decay,
            ),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        return self.lam(x)


class TransitionBase(SerialisableModule):
    """Shared trunk for both transition models: embeds history, conditions on action."""

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
        horizon: int = 1,
        patch_size: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.vocab_size = vocab_size
        self.patch_size = patch_size

        self.tokeniser = ScreenTokeniser()
        self.embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        self.S = self.embed.n_tokens

        self.action_proj = nn.Linear(latent_dim, d_model, bias=bias)
        self.goal_proj = (
            nn.Linear(option_dim, d_model, bias=bias) if option_dim is not None else None
        )
        self.trunk = SpatioTemporalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=self.S, max_temporal_len=context_length + horizon - 1,
            dropout=dropout, bias=bias, causal_temporal=True,
        )
        self.ln_trunk = LayerNorm(d_model, bias)

    def _cond(self, action: torch.Tensor, option_code: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.action_proj(action)
        if option_code is not None and self.goal_proj is not None:
            c = c + self.goal_proj(option_code)
        return c.view(action.shape[0], 1, 1, self.d_model)

    def _encode(self, history: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Embed tokenised history and add action conditioning."""
        return self.ln_trunk(self.trunk(self.embed(history) + cond))


class ObservableTransitionModel(TransitionBase):
    """GENIE-style transition model: predicts next pixel observation(s).

    action is broadcast-added to all history embeddings (GENIE-style).
    Supports single-step and multi-step (teacher-forced or autoregressive) prediction.
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
        super().__init__(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, latent_dim=latent_dim,
            option_dim=option_dim, horizon=horizon,
            patch_size=patch_size, dropout=dropout, bias=bias,
        )
        self.predict_sequence = predict_sequence
        self.horizon = horizon
        self.state_head = nn.Linear(d_model, vocab_size * patch_size**2, bias=bias)

    def _to_logits(self, hid: torch.Tensor) -> torch.Tensor:
        P = self.patch_size
        logits = self.state_head(hid)
        if P == 1:
            return logits
        H, W, V = self.obs_h, self.obs_w, self.vocab_size
        prefix = logits.shape[:-2]
        flat = logits.reshape(-1, H // P, W // P, P, P, V)
        flat = flat.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, H * W, V)
        return flat.reshape(*prefix, H * W, V)

    def forward(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        option_code: Optional[torch.Tensor] = None,
        horizon: int = 1,
        teacher_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, c = history.shape[:2]
        history = self.tokeniser(history)
        cond = self._cond(action, option_code)

        if self.predict_sequence:
            if teacher_frames is not None:
                teacher_frames = self.tokeniser(teacher_frames)
                inp = torch.cat([history, teacher_frames[:, :-1]], dim=1)
                hid = self._encode(inp, cond)
                return self._to_logits(hid[:, c - 1 : c + horizon - 1])
            else:
                frames, current = [], history
                for _ in range(horizon):
                    hid = self._encode(current, cond)
                    logits = self._to_logits(hid[:, -1])
                    frames.append(logits)
                    next_f = logits.argmax(dim=-1).reshape(B, 1, self.obs_h, self.obs_w)
                    current = torch.cat([current[:, 1:], next_f], dim=1)
                return torch.stack(frames, dim=1)

        hid = self._encode(history, cond)
        return self._to_logits(hid[:, -1])


class LatentTransitionModel(TransitionBase):
    """JEPA-style transition model: predicts next latent representation.

    action is broadcast-added to all history embeddings.
    Predicts a single target latent vector (no sequence unrolling).
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
        target_dim: int,
        patch_size: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, latent_dim=latent_dim,
            horizon=1, patch_size=patch_size, dropout=dropout, bias=bias,
        )
        self.latent_head = nn.Linear(d_model, target_dim, bias=bias)
        self.ln_latent = LayerNorm(target_dim, bias)

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        history = self.tokeniser(history)
        cond = self._cond(action, None)
        hid = self._encode(history, cond)
        return self.ln_latent(self.latent_head(hid[:, -1].mean(dim=1)))
