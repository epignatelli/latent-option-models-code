"""Latent Action Model, Observable Transition Model, and Latent Transition Model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .modules import LayerNorm, PatchEmbedding, SpatioTemporalTransformer, VectorQuantizer
from .modules import SerialisableModule
from .tokeniser import ScreenTokeniser


class LatentActionModel(SerialisableModule):
    """VQ bottleneck that maps encoder output to a discrete latent code.

    Applies a linear projection, layer normalisation, and vector quantisation
    in sequence.  Takes a pre-pooled encoder representation and returns the
    quantised vector, a dictionary of VQ losses, and the codebook index.

    Args:
        in_dim: dimensionality of the input encoder representation.
        latent_dim: dimensionality of the quantised code space.
        num_options: codebook size ``K`` (number of discrete options/actions).
        bias: if ``True``, adds bias to the linear projection. Default: ``False``.
        vq_dropout: dropout on the VQ distance matrix. Default: ``0.1``.
        vq_entropy_weight: weight of the entropy regularisation term.
            Default: ``0.01``.
        vq_beta: weight of the commitment loss. Default: ``0.25``.
        vq_reset_thresh: consecutive inactive steps before a dead code is reset.
            Default: ``100``.
        vq_ema_decay: EMA decay for codebook updates. Default: ``0.99``.

    Shape:
        - Input: ``(N, in_dim)``
        - Output ``z_q``: ``(N, latent_dim)``
        - Output ``indices``: ``(N,)``
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
        """
        Args:
            x: encoder pooling of shape ``(N, in_dim)``.

        Returns:
            Tuple of ``(z_q, loss_dict, indices)`` — see
            :class:`~lom.modules.VectorQuantizer` for details.
        """
        return self.lam(x)


class TransitionBase(SerialisableModule):
    """Shared trunk for causal transition models.

    Tokenises and embeds a history of observations, adds a single conditioning
    code (broadcast over all time steps and spatial positions following the
    GENIE formulation), and runs a causal :class:`SpatioTemporalTransformer`.

    This class is not used directly — subclass
    :class:`ObservableTransitionModel` or :class:`LatentTransitionModel`
    instead.

    Args:
        vocab_size: size of the token vocabulary.
        obs_h: observation height in characters.
        obs_w: observation width in characters.
        d_model: transformer embedding dimension.
        n_layers: number of transformer blocks.
        n_heads: number of attention heads.
        context_length: number of history frames.
        latent_dim: dimensionality of the conditioning code vector.
        horizon: number of future steps the transformer must support.
            Affects the temporal positional encoding capacity. Default: ``1``.
        patch_size: spatial patch size. Default: ``1``.
        dropout: dropout probability. Default: ``0.1``.
        bias: if ``True``, adds bias to all linear layers. Default: ``False``.

    Shape:
        - Input ``history``: ``(B, context_length, H, W, 2)``
        - Output of :meth:`encode`: ``(B, context_length, S, d_model)``
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
    ):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.vocab_size = vocab_size
        self.patch_size = patch_size

        self.tokeniser = ScreenTokeniser()
        self.patch_embedding = PatchEmbedding(vocab_size, d_model, obs_h, obs_w, patch_size, bias)
        self.S = self.patch_embedding.n_tokens

        self.code_proj = nn.Linear(latent_dim, d_model, bias=bias)
        self.trunk = SpatioTemporalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_spatial_positions=self.S,
            max_temporal_len=context_length + horizon - 1,
            dropout=dropout,
            bias=bias,
            causal=True,
        )
        self.ln_trunk = LayerNorm(d_model, bias)

    def build_conditioning(self, code: torch.Tensor) -> torch.Tensor:
        """Project a code vector to a ``(B, 1, 1, d_model)`` conditioning tensor.

        The output is shaped for broadcasting over the full ``(B, T, S, D)``
        embedding tensor, so the code is added uniformly to all time steps and
        spatial positions.

        Args:
            code: latent code of shape ``(B, latent_dim)``.

        Returns:
            Conditioning tensor of shape ``(B, 1, 1, d_model)``.
        """
        return self.code_proj(code).view(code.shape[0], 1, 1, self.d_model)

    def encode(self, history: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Embed tokenised history, add conditioning, and run the trunk.

        Args:
            history: tokenised history of shape
                ``(B, T, H, W)`` — long token IDs.
            cond: conditioning tensor of shape ``(B, 1, 1, d_model)``
                as returned by :meth:`build_conditioning`.

        Returns:
            Hidden states of shape ``(B, T, S, d_model)``.
        """
        return self.ln_trunk(self.trunk(self.patch_embedding(history) + cond))


class ObservableTransitionModel(TransitionBase):
    """GENIE-style transition model that predicts next pixel observations.

    Conditions the causal :class:`TransitionBase` trunk on a single latent
    code (action ``z_act`` for LAM, or option ``z_opt`` for LOM), then
    decodes the last hidden state to per-token logits over the vocabulary.

    Supports three prediction modes:

    - **Single-step** (``predict_sequence=False``): predicts the next frame
      from the current context.
    - **Teacher-forced sequence** (``predict_sequence=True``,
      ``teacher_frames`` provided): predicts all ``horizon`` frames in a
      single forward pass using ground-truth frames as decoder input.
    - **Autoregressive sequence** (``predict_sequence=True``,
      no ``teacher_frames``): generates frames one at a time by feeding
      the argmax of each step back as the next input.

    Args:
        vocab_size: size of the token vocabulary.
        obs_h: observation height in characters.
        obs_w: observation width in characters.
        d_model: transformer embedding dimension.
        n_layers: number of transformer blocks.
        n_heads: number of attention heads.
        context_length: number of history frames.
        latent_dim: dimensionality of the conditioning code.
        predict_sequence: if ``True``, enables multi-step prediction.
            Default: ``False``.
        horizon: number of steps to predict when ``predict_sequence=True``.
            Default: ``1``.
        patch_size: spatial patch size. Default: ``1``.
        dropout: dropout probability. Default: ``0.1``.
        bias: if ``True``, adds bias to all linear layers. Default: ``False``.

    Shape:
        - Input ``history``: ``(B, context_length, H, W, 2)``
        - Input ``code``: ``(B, latent_dim)``
        - Output (single-step): ``(B, S, vocab_size)``
        - Output (sequence): ``(B, horizon, S, vocab_size)``
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
        predict_sequence: bool = False,
        horizon: int = 1,
        patch_size: int = 1,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__(
            vocab_size=vocab_size, obs_h=obs_h, obs_w=obs_w,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context_length=context_length, latent_dim=latent_dim,
            horizon=horizon, patch_size=patch_size, dropout=dropout, bias=bias,
        )
        self.predict_sequence = predict_sequence
        self.horizon = horizon
        self.state_head = nn.Linear(d_model, vocab_size * patch_size**2, bias=bias)

    def to_logits(self, hid: torch.Tensor) -> torch.Tensor:
        """Project hidden states to per-token vocabulary logits.

        When ``patch_size > 1``, the patch logits are unpacked back to the
        original ``(H, W)`` token grid.

        Args:
            hid: hidden states of shape ``(B, S, d_model)`` or
                ``(B, horizon, S, d_model)``.

        Returns:
            Logits of shape ``(B, S, vocab_size)`` or
            ``(B, horizon, S, vocab_size)``.
        """
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
        code: torch.Tensor,
        horizon: int = 1,
        teacher_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history: observation history of shape
                ``(B, context_length, H, W, 2)``.
            code: latent conditioning code of shape ``(B, latent_dim)``.
            horizon: number of steps to predict. Ignored when
                ``predict_sequence=False``. Default: ``1``.
            teacher_frames: ground-truth future frames of shape
                ``(B, horizon, H, W, 2)`` for teacher-forced decoding.
                ``None`` uses autoregressive decoding.

        Returns:
            Predicted logits. Shape ``(B, S, vocab_size)`` for single-step
            prediction; ``(B, horizon, S, vocab_size)`` for sequence prediction.
        """
        B, c = history.shape[:2]
        history = self.tokeniser(history)
        cond = self.build_conditioning(code)

        if self.predict_sequence:
            if teacher_frames is not None:
                teacher_frames = self.tokeniser(teacher_frames)
                inp = torch.cat([history, teacher_frames[:, :-1]], dim=1)
                hid = self.encode(inp, cond)
                return self.to_logits(hid[:, c - 1 : c + horizon - 1])
            else:
                frames, current = [], history
                for _ in range(horizon):
                    hid = self.encode(current, cond)
                    logits = self.to_logits(hid[:, -1])
                    frames.append(logits)
                    next_f = logits.argmax(dim=-1).reshape(B, 1, self.obs_h, self.obs_w)
                    current = torch.cat([current[:, 1:], next_f], dim=1)
                return torch.stack(frames, dim=1)

        hid = self.encode(history, cond)
        return self.to_logits(hid[:, -1])


class LatentTransitionModel(TransitionBase):
    """JEPA-style transition model that predicts the next latent representation.

    Conditions the causal :class:`TransitionBase` trunk on a single latent
    code (action ``z_act`` for LAM, or option ``z_opt`` for LOM), then
    projects the last hidden state to a target latent vector.  The prediction
    is trained against the EMA encoder output via cosine distance (JEPA loss),
    so no pixel reconstruction is required.

    Single-step only — sequence unrolling is not needed because the target is
    a single latent vector, not a pixel sequence.

    Args:
        vocab_size: size of the token vocabulary (needed to tokenise history).
        obs_h: observation height in characters.
        obs_w: observation width in characters.
        d_model: transformer embedding dimension.
        n_layers: number of transformer blocks.
        n_heads: number of attention heads.
        context_length: number of history frames.
        latent_dim: dimensionality of the conditioning code.
        target_dim: dimensionality of the predicted latent target.
        patch_size: spatial patch size. Default: ``1``.
        dropout: dropout probability. Default: ``0.1``.
        bias: if ``True``, adds bias to all linear layers. Default: ``False``.

    Shape:
        - Input ``history``: ``(B, context_length, H, W, 2)``
        - Input ``code``: ``(B, latent_dim)``
        - Output: ``(B, target_dim)``
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
        dropout: float = 0.1,
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

    def forward(self, history: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: observation history of shape
                ``(B, context_length, H, W, 2)``.
            code: latent conditioning code of shape ``(B, latent_dim)``.

        Returns:
            Predicted latent representation of shape ``(B, target_dim)``.
        """
        history = self.tokeniser(history)
        cond = self.build_conditioning(code)
        hid = self.encode(history, cond)
        return self.ln_latent(self.latent_head(hid[:, -1].mean(dim=1)))
