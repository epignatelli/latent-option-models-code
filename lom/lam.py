"""Latent Action Model and Dynamics Model."""

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
        self.proj = nn.Linear(in_dim, latent_dim, bias=bias)
        self.ln = LayerNorm(latent_dim, bias)
        self.vq = VectorQuantizer(
            latent_dim=latent_dim,
            num_options=num_options,
            dropout=vq_dropout,
            entropy_weight=vq_entropy_weight,
            vq_beta=vq_beta,
            vq_reset_thresh=vq_reset_thresh,
            ema_decay=vq_ema_decay,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        return self.vq(self.ln(self.proj(x)))


class DynamicsModel(SerialisableModule):
    """Causal STP-Transformer: (history, action[, option_code]) → frame(s) or latent(s).

    action is broadcast-added to all input embeddings as the primary conditioning.
    option_code is an optional secondary conditioning (z_option for LOM dynamics).
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
            self.latent_head = nn.Linear(d_model, target_dim, bias=bias)
            self.ln_latent = LayerNorm(target_dim, bias)
        else:
            self.state_head = nn.Linear(d_model, vocab_size * patch_size**2, bias=bias)

    def _cond(self, action: torch.Tensor, option_code: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.action_proj(action)
        if option_code is not None and self.goal_proj is not None:
            c = c + self.goal_proj(option_code)
        return c.view(action.shape[0], 1, 1, self.d_model)

    def _unpatch_logits(self, logits: torch.Tensor) -> torch.Tensor:
        P = self.patch_size
        if P == 1:
            return logits
        H, W, V = self.obs_h, self.obs_w, self.vocab_size
        prefix = logits.shape[:-2]
        flat = logits.reshape(-1, H // P, W // P, P, P, V)
        flat = flat.permute(0, 1, 3, 2, 4, 5).contiguous()
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
        B, c = history.shape[:2]
        history = self.tokeniser(history)
        if teacher_frames is not None:
            teacher_frames = self.tokeniser(teacher_frames)
        cond = self._cond(action, option_code)

        if self.predict_sequence:
            if teacher_frames is not None:
                inp = torch.cat([history, teacher_frames[:, :-1]], dim=1)
                emb = self.embed(inp) + cond
                hid = self.ln_trunk(self.trunk(emb))
                hid_seq = hid[:, c - 1 : c + horizon - 1]
                if self.predict_latent:
                    return self.ln_latent(self.latent_head(hid_seq.mean(dim=2)))
                return self._unpatch_logits(self.state_head(hid_seq))
            else:
                frames, current = [], history
                for _ in range(horizon):
                    emb = self.embed(current) + cond
                    hid = self.ln_trunk(self.trunk(emb))
                    if self.predict_latent:
                        frames.append(self.ln_latent(self.latent_head(hid[:, -1].mean(dim=1))))
                    else:
                        logits = self._unpatch_logits(self.state_head(hid[:, -1]))
                        frames.append(logits)
                        next_f = logits.argmax(dim=-1).reshape(B, 1, self.obs_h, self.obs_w)
                        current = torch.cat([current[:, 1:], next_f], dim=1)
                return torch.stack(frames, dim=1)

        emb = self.embed(history) + cond
        hid = self.ln_trunk(self.trunk(emb))
        if self.predict_latent:
            return self.ln_latent(self.latent_head(hid[:, -1].mean(dim=1)))
        return self._unpatch_logits(self.state_head(hid[:, -1]))
