"""Encoder modules: STTEncoder, JEPAEncoder, EMAEncoder."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn

from .modules import (
    LayerNorm,
    PatchEmbedding,
    SpatioTemporalTransformer,
    get_opt_block_mask,
)
from .tokeniser import ScreenTokeniser


class STTEncoder(nn.Module):
    """Single-pass bidirectional encoder: sequence → (B, d_model).

    Embeds the full sequence once, splits at context_length, inserts an OPT
    token between history and future, and runs a bidirectional
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

        self.tokeniser = ScreenTokeniser()
        self.embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        S = self.S = self.embed.n_tokens

        # bridges latent_dim → d_model when the two differ (e.g. 512 → 256)
        self.cond_proj = (
            nn.Linear(condition_dim, d_model, bias=bias) if condition_dim is not None else None
        )
        self.opt_token = nn.Parameter(torch.randn(1, 1, S, d_model) * 0.02)
        self.transformer = SpatioTemporalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_spatial_positions=S,
            max_temporal_len=context_length + 1 + horizon,
            dropout=dropout,
            bias=bias,
            causal_temporal=False,
        )

    @property
    def out_dim(self) -> int:
        return self.d_model

    def forward(
        self,
        sequence: torch.Tensor,       # (B, context_length + horizon, H, W, 2)
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, c = sequence.shape[0], self.context_length
        emb = self.embed(self.tokeniser(sequence))   # (B, c+k, S, D) — one tokenise + embed
        k = emb.shape[1] - c

        hist_emb = emb[:, :c]
        fut_emb  = emb[:, c:]
        opt_emb  = self.opt_token.expand(B, 1, self.S, self.d_model)

        if condition is not None and self.cond_proj is not None:
            fut_emb = fut_emb + self.cond_proj(condition).view(B, 1, 1, self.d_model)
        seq = torch.cat([hist_emb, opt_emb, fut_emb], dim=1)

        hidden = self.transformer(seq, temporal_mask=get_opt_block_mask(c + 1 + k, c, seq.device))
        return hidden[:, c, :, :].mean(dim=1)  # (B, D)


class JEPAEncoder(nn.Module):
    """JEPA encoder: past and future encoded by separate causal transformers.

    past_encoder:   history → last-frame spatial mean → (B, D)
    future_encoder: future  → last-frame spatial mean → (B, D)
    Returns concat of both → (B, 2 * d_model).

    No shared weights; no cross-attention. Condition is injected at the VQ
    level in LatentActionModel, not here.
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
        **_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.context_length = context_length

        self.tokeniser = ScreenTokeniser()
        self.embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        S = self.S = self.embed.n_tokens

        self.past_encoder = SpatioTemporalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_spatial_positions=S,
            max_temporal_len=context_length,
            dropout=dropout,
            bias=bias,
            causal_temporal=True,
        )
        self.future_encoder = SpatioTemporalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_spatial_positions=S,
            max_temporal_len=horizon,
            dropout=dropout,
            bias=bias,
            causal_temporal=True,
        )
        self.proj_target = nn.Linear(d_model, latent_dim, bias=bias)
        self.ln_target = LayerNorm(latent_dim, bias)

    @property
    def out_dim(self) -> int:
        return 2 * self.d_model

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        tokens = self.tokeniser(frames)
        if tokens.ndim == 3:
            tokens = tokens.unsqueeze(1)
        emb = self.embed(tokens)
        out = self.future_encoder(emb)
        pooled = out[:, -1, :, :].mean(dim=1)
        return self.ln_target(self.proj_target(pooled))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # (B, context_length + horizon, H, W, 2) — one tokenise + embed
        emb = self.embed(self.tokeniser(sequence))   # (B, c+k, S, D)
        c = self.context_length
        ctx_pooled = self.past_encoder(emb[:, :c])[:, -1, :, :].mean(dim=1)
        tgt_pooled = self.future_encoder(emb[:, c:])[:, -1, :, :].mean(dim=1)
        return torch.cat([ctx_pooled, tgt_pooled], dim=-1)


class EMAEncoder(nn.Module):
    """EMA shadow of any encoder module; provides stable prediction targets.

    Parameters are not trained; updated each step via momentum average of the
    online encoder.
    """

    def __init__(self, base: nn.Module, decay: float = 0.996):
        super().__init__()
        self.encoder = deepcopy(base)
        self.decay = decay
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        for e, o in zip(self.encoder.parameters(), online.parameters()):
            e.data.mul_(self.decay).add_(o.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> torch.Tensor:
        return self.encoder.encode(*args, **kwargs)
