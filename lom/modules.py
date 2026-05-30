"""Primitives only: low-level building blocks shared across encoders and transition models.

Nothing here is a full model or encoder. Contents:
  - Block mask helpers (causal, OPT-token)
  - SerialisableModule, LayerNorm, MLP
  - BidirectionalAttention / CausalAttention
  - PatchEmbedding
  - SpatioTemporalBlock / SpatioTemporalTransformer
  - VectorQuantizer
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention


# --------------------------------------------------------------------------- #
# --- Block mask helpers ---------------------------------------------------- #
# --------------------------------------------------------------------------- #

causal_block_mask_cache: dict[tuple, BlockMask] = {}
opt_block_mask_cache: dict[tuple, BlockMask] = {}


def causal_mask_cache(T: int, device: torch.device) -> BlockMask:
    """Return a cached causal :class:`BlockMask` of shape ``(T, T)``.

    The mask allows position ``q`` to attend only to positions ``kv <= q``
    (lower-triangular), implementing standard autoregressive attention.
    Masks are cached by ``(T, device)`` so repeated calls with the same
    arguments are free.

    Args:
        T: sequence length.
        device: device on which the mask will be used.

    Returns:
        A :class:`BlockMask` suitable for passing to :func:`flex_attention`.
    """
    key = (T, str(device))
    if key not in causal_block_mask_cache:
        causal_block_mask_cache[key] = create_block_mask(
            lambda b, h, q, kv: q >= kv,
            B=None, H=None, Q_LEN=T, KV_LEN=T, device=device,
        )
    return causal_block_mask_cache[key]


def bidirectional_mask_cache(T: int, opt_pos: int, device: torch.device) -> BlockMask:
    """Return a cached :class:`BlockMask` that prevents the OPT token from
    attending to future frames.

    All positions attend bidirectionally except position ``opt_pos`` (the OPT
    token), which is blocked from attending to any position ``kv > opt_pos``.
    This lets OPT summarise history without leaking information from future
    frames, as required by :class:`STTEncoder`.

    Args:
        T: total sequence length (context + 1 OPT token + horizon).
        opt_pos: temporal index of the OPT token (equals ``context_length``).
        device: device on which the mask will be used.

    Returns:
        A :class:`BlockMask` suitable for passing to :func:`flex_attention`.
    """
    key = (T, opt_pos, str(device))
    if key not in opt_block_mask_cache:
        def mask_mod(b, h, q_idx, kv_idx):
            return ~((q_idx == opt_pos) & (kv_idx > opt_pos))
        opt_block_mask_cache[key] = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device,
        )
    return opt_block_mask_cache[key]


# --------------------------------------------------------------------------- #
# --- Base ------------------------------------------------------------------ #
# --------------------------------------------------------------------------- #


class SerialisableModule(nn.Module):
    """Base class adding a :meth:`num_parameters` convenience method."""

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# --- Primitives ------------------------------------------------------------ #
# --------------------------------------------------------------------------- #


class LayerNorm(nn.Module):
    """Layer normalisation with optional bias.

    The PyTorch built-in :class:`torch.nn.LayerNorm` requires a bias term;
    this variant allows ``bias=False`` to reduce parameter count, following
    the GPT-2 ablation by Biewald 2022.

    Args:
        ndim: number of features to normalise.
        bias: if ``True``, adds a learnable bias. Default: ``False``.

    Shape:
        - Input: ``(*, ndim)``
        - Output: ``(*, ndim)``
    """

    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    """Position-wise two-layer feed-forward network with GELU activation.

    Expands to ``4 * d_model`` hidden units then projects back, following
    the standard Transformer MLP block. Dropout is applied after each
    linear projection.

    Args:
        d_model: input and output feature dimension.
        dropout: dropout probability applied after each linear layer.
            Default: ``0.1``.
        bias: if ``True``, adds bias to linear layers. Default: ``False``.

    Shape:
        - Input: ``(*, d_model)``
        - Output: ``(*, d_model)``
    """

    def __init__(self, d_model: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# --------------------------------------------------------------------------- #
# --- Attention ------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class BidirectionalAttention(nn.Module):
    """Multi-head self-attention with full (non-causal) visibility.

    Uses :func:`torch.nn.attention.flex_attention.flex_attention` for all
    sequence lengths. The kernel is forced to the regular flex_attention path
    (not the decoding path) to avoid a PyTorch bug with multi-block masks on
    short sequences (pytorch#147267).

    Subclass :class:`CausalAttention` provides a causal variant.

    Args:
        d_model: total embedding dimension. Must be divisible by ``n_heads``.
        n_heads: number of attention heads.
        dropout: dropout probability on the output projection. Default: ``0.1``.
        bias: if ``True``, adds bias to QKV and output projections.
            Default: ``False``.

    Shape:
        - Input: ``(B, T, d_model)``
        - Output: ``(B, T, d_model)``
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_drop = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.d_model = d_model

    def attend(self, x: torch.Tensor, block_mask: BlockMask | None = None) -> torch.Tensor:
        """Run scaled dot-product attention with an optional structural mask.

        Args:
            x: input tensor of shape ``(B, T, d_model)``.
            block_mask: optional :class:`BlockMask` for structural sparsity
                (e.g. causal or OPT-token masking). ``None`` means full
                bidirectional attention.

        Returns:
            Output tensor of shape ``(B, T, d_model)``.
        """
        B, T, C = x.shape
        head_dim = C // self.n_heads
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, head_dim).transpose(1, 2)
        # T < 128 triggers the flex_decoding kernel which fails for H>1 block masks
        # (pytorch#147267). Force the regular flex_attention kernel unconditionally.
        y = flex_attention(
            q, k, v, block_mask=block_mask, kernel_options={"FORCE_USE_FLEX_ATTENTION": True}
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attend(x, None)


class CausalAttention(BidirectionalAttention):
    """Multi-head self-attention with an auto-built causal mask (decoder-only).

    Each position attends only to itself and earlier positions.  The causal
    :class:`BlockMask` is cached by sequence length and device, so the first
    call for a given ``(T, device)`` pair pays the creation cost; all
    subsequent calls are free.

    Inherits all constructor arguments from :class:`BidirectionalAttention`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attend(x, causal_mask_cache(x.shape[1], x.device))


# --------------------------------------------------------------------------- #
# --- Patch Embedding ------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class PatchEmbedding(nn.Module):
    """Embeds tokenised NetHack screen observations into patch tokens.

    Each frame ``(H, W)`` of integer token IDs is divided into non-overlapping
    ``patch_size × patch_size`` patches. Each patch is embedded via a shared
    character embedding table and then linearly projected to ``d_model``
    dimensions, producing ``n_tokens = (H // P) * (W // P)`` tokens per frame.
    With ``patch_size=1`` the projection is skipped and each screen cell
    becomes one token.

    A ``token_usage`` buffer (shape ``(vocab_size,)``) counts how many times
    each token ID has been seen across all forward passes.  Use it to identify
    dead codebook entries after training::

        dead = (model.embed.token_usage == 0).nonzero()

    .. note::
        :class:`torch.nn.Embedding` is not covered by PyTorch autocast.
        When autocast is enabled, the embedding output is manually cast to
        ``bfloat16`` so that downstream layers see a consistent dtype.

    Args:
        vocab_size: number of distinct token IDs (``CHAR_VOCAB * COLOR_VOCAB``).
        d_model: output embedding dimension.
        obs_h: screen height in characters.
        obs_w: screen width in characters.
        patch_size: spatial patch size ``P``. Both ``obs_h`` and ``obs_w``
            must be divisible by ``P``. Default: ``1``.
        bias: if ``True``, adds bias to the patch projection. Default: ``False``.

    Shape:
        - Input: ``(B, T, H, W)`` — long tensor of token IDs.
        - Output: ``(B, T, n_tokens, d_model)`` where
          ``n_tokens = (H // P) * (W // P)``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        obs_h: int,
        obs_w: int,
        patch_size: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert (
            obs_h % patch_size == 0 and obs_w % patch_size == 0
        ), f"obs_h={obs_h} and obs_w={obs_w} must both be divisible by patch_size={patch_size}"
        self.patch_size = patch_size
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.d_model = d_model
        self.n_tokens = (obs_h // patch_size) * (obs_w // patch_size)

        self.char_embed = nn.Embedding(vocab_size, d_model)
        self.patch_proj = (
            nn.Linear(patch_size**2 * d_model, d_model, bias=bias) if patch_size > 1 else None
        )

        self.token_usage: torch.Tensor
        self.register_buffer("token_usage", torch.zeros(vocab_size, dtype=torch.long))
        self.char_embed.register_forward_hook(self.usage_hook)

    def usage_hook(self, _module, inputs, _output) -> None:
        ids = inputs[0].detach().reshape(-1)
        self.token_usage.index_add_(
            0, ids, torch.ones(ids.numel(), dtype=torch.long, device=ids.device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape
        P, D = self.patch_size, self.d_model

        emb = self.char_embed(x)
        if torch.is_autocast_enabled():
            emb = emb.to(torch.bfloat16)  # nn.Embedding is not autocasted; cast manually

        if self.patch_proj is not None:
            emb = emb.reshape(B, T, H // P, P, W // P, P, D)
            emb = emb.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # (B, T, H/P, W/P, P, P, D)
            emb = emb.reshape(B, T, self.n_tokens, P * P * D)
            emb = self.patch_proj(emb)  # (B, T, n_tokens, D)
        else:
            emb = emb.reshape(B, T, self.n_tokens, D)

        return emb


# --------------------------------------------------------------------------- #
# --- Spatio-Temporal Transformer ------------------------------------------- #
# --------------------------------------------------------------------------- #


class SpatioTemporalBlock(nn.Module):
    """Factored space-time Transformer block.

    Applies three sub-layers in sequence:

    1. **Spatial attention** — each time step attends over its ``S`` spatial
       positions, always bidirectionally.
    2. **Temporal attention** — each spatial position attends over the ``T``
       time steps; causal or bidirectional depending on ``causal``.
    3. **MLP** — position-wise feed-forward network.

    Each sub-layer is preceded by layer normalisation and followed by a
    residual connection (Pre-LN formulation).

    Args:
        d_model: embedding dimension.
        n_heads: number of attention heads.
        dropout: dropout probability. Default: ``0.1``.
        bias: if ``True``, adds bias to all linear layers. Default: ``False``.
        causal: if ``True``, temporal attention uses a causal mask.
            Default: ``False``.

    Shape:
        - Input: ``(B, T, S, D)``
        - Output: ``(B, T, S, D)``
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.ln_s = LayerNorm(d_model, bias)
        self.ln_t = LayerNorm(d_model, bias)
        self.ln_m = LayerNorm(d_model, bias)
        self.spatial_attn = BidirectionalAttention(d_model, n_heads, dropout, bias)
        self.temporal_attn = (
            CausalAttention(d_model, n_heads, dropout, bias)
            if causal
            else BidirectionalAttention(d_model, n_heads, dropout, bias)
        )
        self.mlp = MLP(d_model, dropout, bias)

    def forward(self, x: torch.Tensor, temporal_mask: BlockMask | None = None) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape ``(B, T, S, D)``.
            temporal_mask: optional :class:`BlockMask` for temporal attention.
                ``None`` applies full attention (or the causal mask if
                ``causal=True`` was set at construction).

        Returns:
            Output tensor of shape ``(B, T, S, D)``.
        """
        B, T, S, D = x.shape

        xs = x.reshape(B * T, S, D)
        xs = xs + self.spatial_attn(self.ln_s(xs))
        x = xs.reshape(B, T, S, D)

        xt = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
        xt = xt + self.temporal_attn.attend(self.ln_t(xt), temporal_mask)
        x = xt.reshape(B, S, T, D).permute(0, 2, 1, 3)

        x = x + self.mlp(self.ln_m(x))
        return x


class SpatioTemporalTransformer(nn.Module):
    """Spatio-Temporal Transformer operating on pre-embedded observation sequences.

    Stacks ``n_layers`` :class:`SpatioTemporalBlock` layers with learned
    spatial and temporal positional encodings.  The embedding table is
    intentionally excluded — callers must embed observations first (e.g. via
    :class:`PatchEmbedding`) so that multiple modules can share embeddings
    without re-computing them.

    Positional encodings are learned embeddings, following Genie (Bruce et al.,
    2024):

    - **Spatial**: ``nn.Embedding(n_spatial_positions, d_model)`` — one entry
      per screen token, fixed by the NetHack screen resolution.
    - **Temporal**: ``nn.Embedding(max_temporal_len, d_model)`` — one entry
      per time step; the actual sequence length at forward time may be shorter.

    Args:
        d_model: embedding dimension.
        n_layers: number of :class:`SpatioTemporalBlock` layers.
        n_heads: number of attention heads per block.
        n_spatial_positions: number of spatial tokens per frame
            (``(H // patch_size) * (W // patch_size)``).
        max_temporal_len: maximum sequence length the temporal positional
            encoding supports.
        dropout: dropout probability. Default: ``0.1``.
        bias: if ``True``, adds bias to all linear layers. Default: ``False``.
        causal: if ``True``, temporal attention in every block is causal.
            Default: ``False``.

    Shape:
        - Input: ``(B, T, S, D)`` — pre-embedded, without positional information.
        - Output: ``(B, T, S, D)``
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_spatial_positions: int,
        max_temporal_len: int,
        dropout: float = 0.1,
        bias: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.spatial_pos = nn.Embedding(n_spatial_positions, d_model)
        self.temporal_pos = nn.Embedding(max_temporal_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SpatioTemporalBlock(d_model, n_heads, dropout, bias, causal) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(d_model, bias)
        self.n_spatial_positions = n_spatial_positions
        self.max_temporal_len = max_temporal_len

    def forward(self, x: torch.Tensor, temporal_mask: BlockMask | None = None) -> torch.Tensor:
        """
        Args:
            x: pre-embedded input of shape ``(B, T, S, D)``.
            temporal_mask: optional :class:`BlockMask` for temporal attention
                across all blocks. ``None`` applies full (or causal) attention.

        Returns:
            Output tensor of shape ``(B, T, S, D)``.
        """
        B, T, S, D = x.shape
        assert T <= self.max_temporal_len, (
            f"Sequence length T={T} exceeds max_temporal_len={self.max_temporal_len}"
        )
        s_idx = torch.arange(S, device=x.device)
        t_idx = torch.arange(T, device=x.device)

        x = x + self.spatial_pos(s_idx)[None, None, :, :]
        x = x + self.temporal_pos(t_idx)[None, :, None, :]
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, temporal_mask=temporal_mask)

        return self.ln_f(x)


# --------------------------------------------------------------------------- #
# --- Vector Quantizer ------------------------------------------------------ #
# --------------------------------------------------------------------------- #


class VectorQuantizer(nn.Module):
    """EMA vector quantiser with cosine-distance assignment.

    Implements the VQ-VAE codebook (van den Oord et al., 2017) with three
    key extensions:

    1. **Cosine-distance assignment** — encoder outputs and codebook entries
       are L2-normalised before computing distances, keeping the codebook on
       the unit sphere.
    2. **EMA codebook updates** — the codebook is a non-trainable buffer
       updated by exponential moving average, decoupling codebook learning
       from the choice of optimiser.
    3. **Dead-code reset** — codes that have not been assigned for
       ``vq_reset_thresh`` consecutive steps are replaced with a randomly
       chosen live code, preventing codebook collapse.
    4. **Entropy regularisation** — a negative entropy term on the soft
       assignment distribution encourages uniform codebook usage.

    Only the commitment loss (encoder output close to the chosen code) flows
    through the optimiser; the codebook itself is updated in-place.

    .. note::
        Dropout is applied to the distance matrix before argmin, acting as
        stochastic codebook regularisation during training.

    Args:
        latent_dim: dimensionality of each code vector.
        num_options: codebook size ``K``.
        dropout: dropout probability applied to the distance matrix.
        entropy_weight: weight of the entropy regularisation term.
        vq_beta: weight of the commitment loss term.
        vq_reset_thresh: number of consecutive inactive steps before a dead
            code is reset. Set to ``0`` to disable. Default: ``100``.
        ema_decay: EMA decay factor for codebook updates. Default: ``0.99``.

    Shape:
        - Input ``z``: ``(N, latent_dim)``
        - Output ``z_q``: ``(N, latent_dim)`` — quantised via straight-through.
        - Output ``indices``: ``(N,)`` — codebook indices.
    """

    def __init__(
        self,
        latent_dim: int,
        num_options: int,
        dropout: float,
        entropy_weight: float,
        vq_beta: float,
        vq_reset_thresh: int = 100,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_options = num_options
        self.entropy_weight = entropy_weight
        self.vq_beta = vq_beta
        self.vq_reset_thresh = vq_reset_thresh
        self.ema_decay = ema_decay
        self.drop = nn.Dropout(dropout)

        bound = (3 / latent_dim) ** 0.5
        codebook_init = F.normalize(
            torch.empty(num_options, latent_dim).uniform_(-bound, bound), dim=-1
        )
        # Codebook is a buffer — updated by EMA, not the optimiser.
        self.register_buffer("codebook", codebook_init)
        self.register_buffer("ema_cluster_size", torch.ones(num_options))
        self.register_buffer("ema_embed_sum", codebook_init.clone())
        self.register_buffer("last_active", torch.zeros(num_options, dtype=torch.int64))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """Quantise a batch of continuous vectors.

        Args:
            z: continuous encoder output of shape ``(N, latent_dim)``.

        Returns:
            Tuple of:

            - ``z_q`` ``(N, latent_dim)``: quantised vectors with
              straight-through gradient estimator.
            - ``loss_dict``: dict containing scalar tensors
              ``vq_loss``, ``commit_loss``, ``entropy``.
            - ``indices`` ``(N,)``: codebook indices of the assigned codes.
        """
        z_norm = F.normalize(z, dim=-1)
        cb_norm = F.normalize(self.codebook, dim=-1)
        dist = -self.drop(torch.matmul(z_norm, cb_norm.T))  # (N, K)

        indices = dist.argmin(dim=-1)
        z_hard = F.normalize(self.codebook[indices], dim=-1)

        if self.training:
            with torch.no_grad():
                one_hot = torch.zeros(z_norm.shape[0], self.num_options, device=z.device)
                one_hot.scatter_(1, indices.unsqueeze(1), 1)

                # EMA update: use float32 to avoid bf16 accumulation under autocast
                z_norm_f32 = z_norm.float()
                cluster_size = one_hot.sum(0)
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    cluster_size, alpha=1 - self.ema_decay
                )
                embed_sum = one_hot.T @ z_norm_f32  # (K, D)
                self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

                # Laplace-smoothed codebook update
                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + 1e-5) / (n + self.num_options * 1e-5) * n
                self.codebook.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

                # Dead-code reset: copy a random live entry into each dead slot
                if self.vq_reset_thresh > 0:
                    self.last_active += 1
                    self.last_active[indices.view(-1).unique()] = 0
                    dead = (self.last_active >= self.vq_reset_thresh).nonzero(as_tuple=True)[0]
                    if dead.numel():
                        alive = (self.last_active < self.vq_reset_thresh).nonzero(as_tuple=True)[0]
                        if alive.numel():
                            src = alive[
                                torch.randint(alive.numel(), (dead.numel(),), device=z.device)
                            ]
                            self.codebook[dead] = self.codebook[src]
                            self.ema_embed_sum[dead] = self.ema_embed_sum[src]
                            self.ema_cluster_size[dead] = self.ema_cluster_size[src]
                            self.last_active[dead] = 0

        z_q = z + (z_hard - z).detach()  # straight-through estimator

        commit_loss = (1 - F.cosine_similarity(z, z_hard.detach(), dim=-1)).mean()
        entropy = self.compute_entropy(dist)
        vq_loss = self.vq_beta * commit_loss - self.entropy_weight * entropy

        return z_q, {"vq_loss": vq_loss, "commit_loss": commit_loss, "entropy": entropy}, indices

    def compute_entropy(self, dist: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """Compute the entropy of the mean soft-assignment distribution.

        A higher entropy indicates more uniform codebook usage. Used as a
        regularisation signal to discourage codebook collapse.

        Args:
            dist: distance matrix of shape ``(N, K)`` (negative cosine
                similarities before argmin).
            eps: small constant for numerical stability. Default: ``1e-9``.

        Returns:
            Scalar entropy tensor.
        """
        avg_probs = F.softmax(-dist, dim=-1).mean(0)  # (K,) mean soft assignment over batch
        return -(avg_probs * avg_probs.clamp(min=eps).log()).sum()

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised codebook entries for the given indices.

        Args:
            indices: long tensor of codebook indices, any shape.

        Returns:
            Normalised code vectors of shape ``(*indices.shape, latent_dim)``.
        """
        return F.normalize(self.codebook[indices], dim=-1)
