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
    bidirectional_mask_cache,
)
from .tokeniser import ScreenTokeniser


class STTEncoder(nn.Module):
    """Spatio-Temporal Transformer encoder (bidirectional, single-pass).

    Encodes a pair of observation sequences — history and future — into a
    single vector by concatenating them around a learned OPT token and running
    a bidirectional :class:`SpatioTemporalTransformer`.  The OPT token is
    masked from attending to future frames, so it summarises the history while
    remaining aware of future context through the attention of future tokens
    back to history.  Its spatial mean is returned as the pooled representation.

    The full sequence passed to the transformer is::

        [history (c frames) | OPT (1 frame) | future (k frames)]

    where ``c = context_length`` and ``k = horizon``.

    When ``condition`` is provided (e.g. ``z_opt`` for the action encoder),
    it is broadcast-added to the future frame embeddings only — following the
    GENIE formulation where the option signal modulates how the observed
    transition is read, without retroactively altering the history.

    Args:
        vocab_size: size of the token vocabulary.
        obs_h: observation height in characters.
        obs_w: observation width in characters.
        d_model: transformer embedding dimension.
        n_layers: number of transformer blocks.
        n_heads: number of attention heads.
        context_length: number of history frames ``c``.
        horizon: number of future frames ``k``. Default: ``1``.
        patch_size: spatial patch size for :class:`PatchEmbedding`.
            Default: ``1``.
        condition_dim: if set, a linear projection maps a conditioning vector
            of this dimension to ``d_model`` and adds it to the future
            embeddings. Default: ``None`` (no conditioning).
        dropout: dropout probability. Default: ``0.1``.
        bias: if ``True``, adds bias to all linear layers. Default: ``False``.

    Shape:
        - Input ``sequence``: ``(B, c + k, H, W, 2)`` — raw screen observations.
        - Input ``condition``: ``(B, condition_dim)`` — optional.
        - Output: ``(B, d_model)``
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
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.horizon = horizon

        self.tokeniser = ScreenTokeniser()
        self.embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        S = self.S = self.embed.n_tokens

        # bridges condition_dim → d_model when the two differ (e.g. 512 → 256)
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
            causal=False,
        )

    @property
    def out_dim(self) -> int:
        """Output dimensionality: ``d_model``."""
        return self.d_model

    def forward(
        self,
        sequence: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence: raw screen observations of shape
                ``(B, context_length + horizon, H, W, 2)``.
            condition: optional conditioning vector of shape
                ``(B, condition_dim)``, broadcast-added to future embeddings.

        Returns:
            Pooled OPT-token representation of shape ``(B, d_model)``.
        """
        B, c = sequence.shape[0], self.context_length
        emb = self.embed(self.tokeniser(sequence))   # (B, c+k, S, D) — one tokenise + embed
        k = emb.shape[1] - c

        hist_emb = emb[:, :c]
        fut_emb  = emb[:, c:]
        opt_emb  = self.opt_token.expand(B, 1, self.S, self.d_model)

        if condition is not None and self.cond_proj is not None:
            fut_emb = fut_emb + self.cond_proj(condition).view(B, 1, 1, self.d_model)
        seq = torch.cat([hist_emb, opt_emb, fut_emb], dim=1)

        hidden = self.transformer(seq, temporal_mask=bidirectional_mask_cache(c + 1 + k, c, seq.device))
        return hidden[:, c, :, :].mean(dim=1)  # (B, D) — OPT is always at position c


class JEPAEncoder(nn.Module):
    """JEPA encoder with separate causal transformers for past and future.

    Encodes history and future through independent causal (decoder-only)
    transformers — no shared weights, no cross-attention.  Each transformer
    outputs the last-frame spatial mean as a compact representation.  The two
    representations are concatenated to form a ``2 * d_model`` vector.

    This design follows JEPA (Assran et al., 2023): the online encoder sees
    both context and target; the EMA target encoder (:class:`EMAEncoder`)
    wraps this module and calls :meth:`encode` to produce stable target
    representations from future frames only.

    .. note::
        Condition signals (e.g. ``z_opt``) are not injected here; they are
        concatenated at the VQ bottleneck input level in
        :class:`~lom.lam.LatentActionModel`.

    Args:
        vocab_size: size of the token vocabulary.
        obs_h: observation height in characters.
        obs_w: observation width in characters.
        d_model: transformer embedding dimension for each encoder.
        n_layers: number of transformer blocks in each encoder.
        n_heads: number of attention heads.
        context_length: number of history frames.
        latent_dim: output dimension of :meth:`encode` (EMA target dimension).
        horizon: number of future frames. Default: ``1``.
        patch_size: spatial patch size. Default: ``1``.
        dropout: dropout probability. Default: ``0.1``.
        bias: if ``True``, adds bias to all linear layers. Default: ``False``.

    Shape:
        - Input ``sequence``: ``(B, context_length + horizon, H, W, 2)``.
        - Output of :meth:`forward`: ``(B, 2 * d_model)``.
        - Output of :meth:`encode`: ``(B, latent_dim)``.
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
        dropout: float = 0.1,
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
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=S, max_temporal_len=context_length,
            dropout=dropout, bias=bias, causal=True,
        )
        self.future_encoder = SpatioTemporalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=S, max_temporal_len=horizon,
            dropout=dropout, bias=bias, causal=True,
        )
        self.proj_target = nn.Linear(d_model, latent_dim, bias=bias)
        self.ln_target = LayerNorm(latent_dim, bias)

    @property
    def out_dim(self) -> int:
        """Output dimensionality of :meth:`forward`: ``2 * d_model``."""
        return 2 * self.d_model

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames through the future encoder only.

        Used by :class:`EMAEncoder` to produce stable JEPA prediction targets
        from future observations without running the past encoder.

        Args:
            frames: raw screen observations of shape ``(B, H, W, 2)`` (single
                frame) or ``(B, k, H, W, 2)`` (sequence).

        Returns:
            Projected representation of shape ``(B, latent_dim)``.
        """
        tokens = self.tokeniser(frames)
        if tokens.ndim == 3:
            tokens = tokens.unsqueeze(1)
        emb = self.embed(tokens)
        out = self.future_encoder(emb)
        pooled = out[:, -1, :, :].mean(dim=1)
        return self.ln_target(self.proj_target(pooled))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Encode a full sequence through both past and future encoders.

        Args:
            sequence: raw screen observations of shape
                ``(B, context_length + horizon, H, W, 2)``.

        Returns:
            Concatenated past and future representations of shape
            ``(B, 2 * d_model)``.
        """
        emb = self.embed(self.tokeniser(sequence))   # (B, c+k, S, D) — one tokenise + embed
        c = self.context_length
        ctx_pooled = self.past_encoder(emb[:, :c])[:, -1, :, :].mean(dim=1)
        tgt_pooled = self.future_encoder(emb[:, c:])[:, -1, :, :].mean(dim=1)
        return torch.cat([ctx_pooled, tgt_pooled], dim=-1)


class EMAEncoder(nn.Module):
    """Exponential moving average shadow of any encoder module.

    Maintains a non-trainable copy of a given encoder whose parameters are
    updated each step as an EMA of the online encoder's parameters::

        ema_param = decay * ema_param + (1 - decay) * online_param

    Provides stable prediction targets for JEPA-style self-supervised
    objectives, preventing representation collapse by decoupling the target
    network from gradient updates.

    Call :meth:`update` after each optimiser step to synchronise.

    Args:
        base: the online encoder module to shadow. A deep copy is taken at
            construction time.
        decay: EMA decay factor. Higher values make the target network
            change more slowly. Default: ``0.996``.
    """

    def __init__(self, base: nn.Module, decay: float = 0.996):
        super().__init__()
        self.encoder = deepcopy(base)
        self.decay = decay
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        """Update the EMA parameters from the online encoder.

        Args:
            online: the online encoder whose parameters are the source.
        """
        for e, o in zip(self.encoder.parameters(), online.parameters()):
            e.data.mul_(self.decay).add_(o.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the EMA encoder."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> torch.Tensor:
        """Call the ``encode`` method of the wrapped encoder."""
        return self.encoder.encode(*args, **kwargs)  # type: ignore[attr-defined]
