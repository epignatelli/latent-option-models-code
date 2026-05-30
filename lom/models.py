"""Top-level LOM model classes.

Two concrete model variants, each self-contained:

  ReconstructionLOM — bidirectional encoder + pixel-level reconstruction dynamics
  LatentLOM         — factored causal encoder + latent-space dynamics (EMA targets)

Both take (history, future) and return a dict of predictions and VQ info.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import EMAEncoder
from .lam import LatentActionModel, ObservableTransitionModel, LatentTransitionModel
from .modules import (
    LayerNorm,
    PatchEmbedding,
    SerialisableModule,
    SpatioTemporalTransformer,
    bidirectional_mask_cache,
)
from .tokeniser import ScreenTokeniser


class ReconstructionLOM(SerialisableModule):
    """LOM with a bidirectional encoder and pixel-level reconstruction dynamics.

    Both the option and action encoders concatenate their input sequence around
    a learned OPT token and run a bidirectional
    :class:`~lom.modules.SpatioTemporalTransformer`.  The OPT token is masked
    from attending to future frames; its spatial mean is the pooled
    representation.  The option code ``z_opt`` is broadcast-added to the action
    encoder's future frames before the transformer, following the GENIE
    formulation.

    Args:
        vocab_size: token vocabulary size.
        obs_h: observation height in characters.
        obs_w: observation width in characters.
        n_actions: action vocabulary size (codebook size for the action VQ).
        d_model: transformer embedding dimension.
        n_layers: number of transformer blocks.
        n_heads: number of attention heads.
        context_length: number of history frames.
        horizon: number of future frames for the option encoder.
        latent_dim: dimensionality of the quantised code space.
        num_options: option codebook size.
        patch_size: spatial patch size. Default: ``1``.
        predict_sequence: if ``True``, LOM dynamics predicts the full horizon.
            Default: ``False``.
        dropout: dropout probability. Default: ``0.1``.
        bias: if ``True``, adds bias to all linear layers. Default: ``False``.
        vq_dropout: VQ distance-matrix dropout. Default: ``0.1``.
        vq_entropy_weight: VQ entropy regularisation weight. Default: ``0.01``.
        vq_beta: VQ commitment loss weight. Default: ``0.25``.
        vq_reset_thresh: VQ dead-code reset threshold. Default: ``100``.
        vq_ema_decay: VQ EMA decay. Default: ``0.99``.
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
        dropout: float = 0.1,
        bias: bool = False,
        vq_dropout: float = 0.1,
        vq_entropy_weight: float = 0.01,
        vq_beta: float = 0.25,
        vq_reset_thresh: int = 100,
        vq_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.context_length = context_length
        self.horizon = horizon
        self.predict_sequence = predict_sequence

        vq_kwargs = dict(
            bias=bias, vq_dropout=vq_dropout, vq_entropy_weight=vq_entropy_weight,
            vq_beta=vq_beta, vq_reset_thresh=vq_reset_thresh, vq_ema_decay=vq_ema_decay,
        )
        self.tokeniser = ScreenTokeniser()

        # --- Option encoder (LOM): bidirectional over [history | OPT | future] ---
        self.opt_embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        S = self.opt_embed.n_tokens
        self.opt_token = nn.Parameter(torch.randn(1, 1, S, d_model) * 0.02)
        self.opt_transformer = SpatioTemporalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=S, max_temporal_len=context_length + 1 + horizon,
            dropout=dropout, bias=bias, causal=False,
        )
        self.opt_vq = LatentActionModel(in_dim=d_model, latent_dim=latent_dim,
                                        num_options=num_options, **vq_kwargs)

        # --- Action encoder (LAM): bidirectional over [history | OPT | next_frame] ---
        # z_opt is broadcast-added to the next_frame embedding before the transformer.
        self.act_embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        self.act_token = nn.Parameter(torch.randn(1, 1, S, d_model) * 0.02)
        self.act_cond_proj = nn.Linear(latent_dim, d_model, bias=bias)  # z_opt → d_model
        self.act_transformer = SpatioTemporalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=S, max_temporal_len=context_length + 1 + 1,
            dropout=dropout, bias=bias, causal=False,
        )
        self.act_vq = LatentActionModel(in_dim=d_model, latent_dim=latent_dim,
                                        num_options=n_actions, **vq_kwargs)

        # --- Dynamics ---
        dyn_kwargs = dict(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, latent_dim=latent_dim,
            patch_size=patch_size, dropout=dropout, bias=bias,
        )
        self.lam_dynamics = ObservableTransitionModel(**dyn_kwargs)
        self.lom_dynamics = ObservableTransitionModel(
            **dyn_kwargs, predict_sequence=predict_sequence, horizon=horizon,
        )

    def encode_option(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Encode history + future into the option representation.

        Args:
            history: ``(B, c, H, W, 2)``
            future: ``(B, k, H, W, 2)``

        Returns:
            OPT-token pooling of shape ``(B, d_model)``.
        """
        B, c = history.shape[:2]
        k = future.shape[1]
        emb = self.opt_embed(self.tokeniser(torch.cat([history, future], dim=1)))
        opt = self.opt_token.expand(B, 1, self.opt_embed.n_tokens, emb.shape[-1])
        seq = torch.cat([emb[:, :c], opt, emb[:, c:]], dim=1)
        hidden = self.opt_transformer(seq, temporal_mask=bidirectional_mask_cache(
            c + 1 + k, c, seq.device))
        return hidden[:, c].mean(dim=1)  # OPT is always at position c

    def encode_action(self, history: torch.Tensor, next_frame: torch.Tensor,
                      z_opt: torch.Tensor) -> torch.Tensor:
        """Encode history + next_frame conditioned on z_opt.

        Args:
            history: ``(B, c, H, W, 2)``
            next_frame: ``(B, 1, H, W, 2)``
            z_opt: ``(B, latent_dim)`` — broadcast-added to next_frame embeddings.

        Returns:
            OPT-token pooling of shape ``(B, d_model)``.
        """
        B, c = history.shape[:2]
        emb = self.act_embed(self.tokeniser(torch.cat([history, next_frame], dim=1)))
        # Condition next_frame embeddings on z_opt
        emb[:, c:] = emb[:, c:] + self.act_cond_proj(z_opt).view(B, 1, 1, emb.shape[-1])
        opt = self.act_token.expand(B, 1, self.act_embed.n_tokens, emb.shape[-1])
        seq = torch.cat([emb[:, :c], opt, emb[:, c:]], dim=1)
        hidden = self.act_transformer(seq, temporal_mask=bidirectional_mask_cache(
            c + 2, c, seq.device))
        return hidden[:, c].mean(dim=1)

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> dict:
        """
        Args:
            history: ``(B, c, H, W, 2)``
            future: ``(B, k, H, W, 2)`` — ``k = horizon``; ``future[:, 0:1]`` is next frame.

        Returns:
            Dict with keys ``lam_logits``, ``lom_logits``, ``z_opt``, ``z_act``,
            ``opt_idx``, ``act_idx``, ``vq_opt``, ``vq_act``.
        """
        z_opt, vq_opt, opt_idx = self.opt_vq(self.encode_option(history, future))

        next_frame = future[:, 0:1]
        z_act, vq_act, act_idx = self.act_vq(
            self.encode_action(history, next_frame, z_opt.detach())
        )

        lam_logits = self.lam_dynamics(history, z_act)
        if self.predict_sequence:
            lom_logits = self.lom_dynamics(history, z_opt, horizon=self.horizon,
                                           teacher_frames=future)
        else:
            lom_logits = self.lom_dynamics(history, z_opt)

        return {
            "lam_logits": lam_logits, "lom_logits": lom_logits,
            "z_opt": z_opt, "z_act": z_act,
            "opt_idx": opt_idx, "act_idx": act_idx,
            "vq_opt": vq_opt, "vq_act": vq_act,
        }


class LatentLOM(SerialisableModule):
    """LOM with factored causal encoders and latent-space dynamics (JEPA targets).

    Each encoder path (option and action) uses two separate causal
    :class:`~lom.modules.SpatioTemporalTransformer` instances — one for history
    (context) and one for future (target) — with no shared weights and no
    cross-attention.  The last-frame spatial means are concatenated and passed
    to the VQ bottleneck.

    EMA copies of the target transformers, projections, and embeddings provide
    stable prediction targets, following the JEPA training objective.

    Args:
        vocab_size: token vocabulary size.
        obs_h: observation height in characters.
        obs_w: observation width in characters.
        n_actions: action vocabulary size.
        d_model: transformer embedding dimension.
        n_layers: number of transformer blocks in each encoder.
        n_heads: number of attention heads.
        context_length: number of history frames.
        horizon: number of future frames for the option encoder.
        latent_dim: dimensionality of the quantised code and EMA target.
        num_options: option codebook size.
        patch_size: spatial patch size. Default: ``1``.
        ema_decay: EMA decay for target network updates. Default: ``0.996``.
        dropout: dropout probability. Default: ``0.1``.
        bias: if ``True``, adds bias to all linear layers. Default: ``False``.
        vq_dropout: VQ distance-matrix dropout. Default: ``0.1``.
        vq_entropy_weight: VQ entropy regularisation weight. Default: ``0.01``.
        vq_beta: VQ commitment loss weight. Default: ``0.25``.
        vq_reset_thresh: VQ dead-code reset threshold. Default: ``100``.
        vq_ema_decay: VQ EMA decay. Default: ``0.99``.
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
        dropout: float = 0.1,
        bias: bool = False,
        vq_dropout: float = 0.1,
        vq_entropy_weight: float = 0.01,
        vq_beta: float = 0.25,
        vq_reset_thresh: int = 100,
        vq_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_dim = latent_dim

        vq_kwargs = dict(
            bias=bias, vq_dropout=vq_dropout, vq_entropy_weight=vq_entropy_weight,
            vq_beta=vq_beta, vq_reset_thresh=vq_reset_thresh, vq_ema_decay=vq_ema_decay,
        )
        self.tokeniser = ScreenTokeniser()

        # --- Option encoder (LOM): separate causal transformers for history and future ---
        self.opt_embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        S = self.opt_embed.n_tokens
        self.opt_context_transformer = SpatioTemporalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=S, max_temporal_len=context_length,
            dropout=dropout, bias=bias, causal=True,
        )
        self.opt_target_transformer = SpatioTemporalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=S, max_temporal_len=horizon,
            dropout=dropout, bias=bias, causal=True,
        )
        self.opt_proj = nn.Linear(d_model, latent_dim, bias=bias)
        self.opt_ln   = LayerNorm(latent_dim, bias)
        self.opt_vq   = LatentActionModel(in_dim=2 * d_model, latent_dim=latent_dim,
                                          num_options=num_options, **vq_kwargs)

        # --- Action encoder (LAM): separate causal transformers for history and next_frame ---
        self.act_embed = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        self.act_context_transformer = SpatioTemporalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=S, max_temporal_len=context_length,
            dropout=dropout, bias=bias, causal=True,
        )
        self.act_target_transformer = SpatioTemporalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_spatial_positions=S, max_temporal_len=1,
            dropout=dropout, bias=bias, causal=True,
        )
        self.act_proj = nn.Linear(d_model, latent_dim, bias=bias)
        self.act_ln   = LayerNorm(latent_dim, bias)
        self.act_vq   = LatentActionModel(in_dim=2 * d_model + latent_dim, latent_dim=latent_dim,
                                          num_options=n_actions, **vq_kwargs)

        # --- EMA target networks (option path) ---
        self.ema_opt_embed              = EMAEncoder(self.opt_embed,              decay=ema_decay)
        self.ema_opt_target_transformer = EMAEncoder(self.opt_target_transformer, decay=ema_decay)
        self.ema_opt_proj               = EMAEncoder(self.opt_proj,               decay=ema_decay)
        self.ema_opt_ln                 = EMAEncoder(self.opt_ln,                 decay=ema_decay)

        # --- EMA target networks (action path) ---
        self.ema_act_embed              = EMAEncoder(self.act_embed,              decay=ema_decay)
        self.ema_act_target_transformer = EMAEncoder(self.act_target_transformer, decay=ema_decay)
        self.ema_act_proj               = EMAEncoder(self.act_proj,               decay=ema_decay)
        self.ema_act_ln                 = EMAEncoder(self.act_ln,                 decay=ema_decay)

        # --- Dynamics ---
        dyn_kwargs = dict(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, latent_dim=latent_dim,
            target_dim=latent_dim, patch_size=patch_size, dropout=dropout, bias=bias,
        )
        self.lam_dynamics = LatentTransitionModel(**dyn_kwargs)
        self.lom_dynamics = LatentTransitionModel(**dyn_kwargs)

    def update_ema(self) -> None:
        """Update all EMA target networks from the current online parameters."""
        self.ema_opt_embed.update(self.opt_embed)
        self.ema_opt_target_transformer.update(self.opt_target_transformer)
        self.ema_opt_proj.update(self.opt_proj)
        self.ema_opt_ln.update(self.opt_ln)
        self.ema_act_embed.update(self.act_embed)
        self.ema_act_target_transformer.update(self.act_target_transformer)
        self.ema_act_proj.update(self.act_proj)
        self.ema_act_ln.update(self.act_ln)

    def encode_option(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Encode history and future into the option representation.

        Returns:
            Concatenated context and target poolings of shape ``(B, 2 * d_model)``.
        """
        c = self.context_length
        emb = self.opt_embed(self.tokeniser(torch.cat([history, future], dim=1)))
        ctx = self.opt_context_transformer(emb[:, :c])[:, -1].mean(dim=1)
        tgt = self.opt_target_transformer(emb[:, c:])[:, -1].mean(dim=1)
        return torch.cat([ctx, tgt], dim=-1)

    def encode_action(self, history: torch.Tensor, next_frame: torch.Tensor) -> torch.Tensor:
        """Encode history and next_frame into the action representation.

        Returns:
            Concatenated context and target poolings of shape ``(B, 2 * d_model)``.
        """
        c = self.context_length
        emb = self.act_embed(self.tokeniser(torch.cat([history, next_frame], dim=1)))
        ctx = self.act_context_transformer(emb[:, :c])[:, -1].mean(dim=1)
        tgt = self.act_target_transformer(emb[:, c:])[:, -1].mean(dim=1)
        return torch.cat([ctx, tgt], dim=-1)

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> dict:
        """
        Args:
            history: ``(B, c, H, W, 2)``
            future: ``(B, k, H, W, 2)`` — ``k = horizon``; ``future[:, 0:1]`` is next frame.

        Returns:
            Dict with keys ``z_act_hat``, ``z_act_target``, ``z_opt_hat``,
            ``z_opt_target``, ``z_opt``, ``z_act``, ``opt_idx``, ``act_idx``,
            ``vq_opt``, ``vq_act``.
        """
        c = self.context_length
        next_frame = future[:, 0:1]

        z_opt, vq_opt, opt_idx = self.opt_vq(self.encode_option(history, future))

        act_repr = self.encode_action(history, next_frame)
        z_act, vq_act, act_idx = self.act_vq(
            torch.cat([act_repr, z_opt.detach()], dim=-1)
        )

        # EMA targets: embed → target transformer → pool → proj → ln
        with torch.no_grad():
            opt_emb_ema = self.ema_opt_embed(self.tokeniser(future))
            z_opt_target = self.ema_opt_ln(self.ema_opt_proj(
                self.ema_opt_target_transformer(opt_emb_ema)[:, -1].mean(dim=1)
            ))

            act_emb_ema = self.ema_act_embed(self.tokeniser(next_frame))
            z_act_target = self.ema_act_ln(self.ema_act_proj(
                self.ema_act_target_transformer(act_emb_ema)[:, -1].mean(dim=1)
            ))

        z_act_hat = self.lam_dynamics(history, z_act)
        z_opt_hat = self.lom_dynamics(history, z_opt)

        return {
            "z_act_hat": z_act_hat, "z_act_target": z_act_target,
            "z_opt_hat": z_opt_hat, "z_opt_target": z_opt_target,
            "z_opt": z_opt, "z_act": z_act,
            "opt_idx": opt_idx, "act_idx": act_idx,
            "vq_opt": vq_opt, "vq_act": vq_act,
        }
