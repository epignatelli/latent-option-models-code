"""Core building blocks: Spatio-Temporal Transformer and Vector Quantizer.

Architecture mirrors the latent-molecule-generation codebase, extended to
handle spatiotemporal observation sequences (T, H, W) via factored
spatial + temporal attention (TimeSformer-style).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention


# --------------------------------------------------------------------------- #
# --- flex_attention helpers ------------------------------------------------ #
# --------------------------------------------------------------------------- #

_causal_block_mask_cache: dict[tuple, BlockMask] = {}
_opt_block_mask_cache: dict[tuple, BlockMask] = {}


def _causal_mask_mod(
    _b: torch.Tensor, _h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
) -> torch.Tensor:
    return q >= kv


def _get_causal_block_mask(T: int, device: torch.device) -> BlockMask:
    key = (T, str(device))
    if key not in _causal_block_mask_cache:
        _causal_block_mask_cache[key] = create_block_mask(
            _causal_mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device
        )
    return _causal_block_mask_cache[key]


def _get_opt_block_mask(T: int, opt_pos: int, device: torch.device) -> BlockMask:
    key = (T, opt_pos, str(device))
    if key not in _opt_block_mask_cache:
        def mask_mod(b, h, q_idx, kv_idx):
            return ~((q_idx == opt_pos) & (kv_idx > opt_pos))
        _opt_block_mask_cache[key] = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device
        )
    return _opt_block_mask_cache[key]


class SerialisableModule(nn.Module):
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (torch built-in doesn't support bias=False)."""

    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SelfAttention(nn.Module):
    """Multi-head self-attention using flex_attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_drop = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.d_model = d_model


    def attend(self, x: torch.Tensor, block_mask: BlockMask | None) -> torch.Tensor:
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
        raise NotImplementedError


class CausalAttention(SelfAttention):
    """Self-attention with a causal mask (decoder-only)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attend(x, _get_causal_block_mask(x.shape[1], x.device))


class BidirectionalAttention(SelfAttention):
    """Full (non-causal) self-attention."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attend(x, None)


class PatchEmbedding(nn.Module):
    """Embeds (B, T, H, W) token IDs into (B, T, n_tokens, d_model) patch tokens.

    Expects pre-tokenised integer IDs (e.g. from ScreenTokeniser).

    patch_size=1  — plain token embedding, no spatial compression.
    patch_size=P  — each P×P block is embedded and projected into one d_model
                    token.  n_tokens = (H//P) * (W//P).

    Registers a `token_usage` buffer (shape: vocab_size,) that counts how many
    times each token ID has been looked up across all forward passes.  Use it
    to identify dead tokens after training:
        dead = (model.embed.token_usage == 0).sum()
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

        self.register_buffer("token_usage", torch.zeros(vocab_size, dtype=torch.long))
        self.char_embed.register_forward_hook(self._usage_hook)

    def _usage_hook(self, _module, inputs, _output) -> None:
        ids = inputs[0].detach().reshape(-1)
        self.token_usage.index_add_(
            0, ids, torch.ones(ids.numel(), dtype=torch.long, device=ids.device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W) long — token IDs
        Returns:
            (B, T, n_tokens, d_model)
        """
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


class SpatioTemporalBlock(nn.Module):
    """Factored space-time attention: spatial first, then temporal, then MLP.

    Spatial attention attends over the H*W spatial positions within each
    time step (always bidirectional).  Temporal attention attends over T
    time steps for each spatial position (causal or bidirectional depending
    on `causal_temporal`).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        causal_temporal: bool = False,
    ):
        super().__init__()
        self.ln_s = LayerNorm(d_model, bias)
        self.ln_t = LayerNorm(d_model, bias)
        self.ln_m = LayerNorm(d_model, bias)
        self.spatial_attn = BidirectionalAttention(d_model, n_heads, dropout, bias)
        self.temporal_attn = (
            CausalAttention(d_model, n_heads, dropout, bias)
            if causal_temporal
            else BidirectionalAttention(d_model, n_heads, dropout, bias)
        )
        self.mlp = MLP(d_model, dropout, bias)

    def forward(self, x: torch.Tensor, temporal_mask: BlockMask | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, S, D)
            temporal_mask: BlockMask for temporal attention, or None for full attention
        Returns:
            (B, T, S, D)
        """
        B, T, S, D = x.shape

        # --- Spatial attention: (B*T, S, D) ---
        xs = x.reshape(B * T, S, D)
        xs = xs + self.spatial_attn(self.ln_s(xs))
        x = xs.reshape(B, T, S, D)

        # --- Temporal attention: (B*S, T, D) ---
        xt = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
        xt = xt + self.temporal_attn.attend(self.ln_t(xt), temporal_mask)
        x = xt.reshape(B, S, T, D).permute(0, 2, 1, 3)

        # --- MLP ---
        x = x + self.mlp(self.ln_m(x))
        return x


class SpatioTemporalTransformer(nn.Module):
    """Spatio-Temporal Transformer over pre-embedded (B, T, S, D) tensors.

    Does not contain a char embedding table — callers embed observations
    before passing them in.

    Positional encoding (both learned, following Genie):
      - Spatial:  nn.Embedding(n_spatial_positions, D) — fixed by the NLE
                  screen size, never changes.
      - Temporal: nn.Embedding(max_temporal_len, D) — capacity parameter;
                  set it to the largest context you will ever need.  The
                  actual sequence length T at forward time can be anything
                  up to max_temporal_len.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_spatial_positions: int,
        max_temporal_len: int,
        dropout: float = 0.0,
        bias: bool = False,
        causal_temporal: bool = False,
    ):
        super().__init__()
        self.spatial_pos = nn.Embedding(n_spatial_positions, d_model)
        self.temporal_pos = nn.Embedding(max_temporal_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(d_model, n_heads, dropout, bias, causal_temporal)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = LayerNorm(d_model, bias)
        self.n_spatial_positions = n_spatial_positions
        self.max_temporal_len = max_temporal_len

    def forward(self, x: torch.Tensor, temporal_mask: BlockMask | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, S, D) — already char-embedded, without positional info
            temporal_mask: optional BlockMask for temporal attention (None = full attention)
        Returns:
            (B, T, S, D)
        """
        B, T, S, D = x.shape
        assert (
            T <= self.max_temporal_len
        ), f"Sequence length T={T} exceeds max_temporal_len={self.max_temporal_len}"
        s_idx = torch.arange(S, device=x.device)
        t_idx = torch.arange(T, device=x.device)

        x = x + self.spatial_pos(s_idx)[None, None, :, :]  # (1,1,S,D)
        x = x + self.temporal_pos(t_idx)[None, :, None, :]  # (1,T,1,D)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, temporal_mask=temporal_mask)

        return self.ln_f(x)



class VectorQuantizer(nn.Module):
    """EMA VQ with cosine-distance assignment, straight-through estimator,
    entropy regularisation and dead-code reset.

    The codebook is a buffer updated by exponential moving average (not the
    optimizer), following van den Oord et al. 2017.  This decouples codebook
    learning from the choice of optimizer and avoids the codebook drifting
    off the unit sphere.

    Only the encoder's commitment loss flows through the optimizer;
    the codebook itself is updated in-place during each forward pass.
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
        # Codebook is a buffer — updated by EMA, not the optimizer.
        self.register_buffer("codebook", codebook_init)
        self.register_buffer("ema_cluster_size", torch.ones(num_options))
        self.register_buffer("ema_embed_sum", codebook_init.clone())
        self.register_buffer("last_active", torch.zeros(num_options, dtype=torch.long))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Args:
            z: (N, latent_dim) — flat batch of continuous vectors
        Returns:
            z_q: (N, latent_dim) quantized with STE
            loss_dict: dict with scalar losses
            indices: (N,) codebook indices
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

        z_q = z + (z_hard - z).detach()  # straight-through

        commit_loss = (1 - F.cosine_similarity(z, z_hard.detach(), dim=-1)).mean()
        entropy = self._entropy(dist)
        vq_loss = self.vq_beta * commit_loss - self.entropy_weight * entropy

        return (
            z_q,
            {
                "vq_loss": vq_loss,
                "commit_loss": commit_loss,
                "entropy": entropy,
            },
            indices,
        )

    def _entropy(self, dist: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        avg_probs = F.softmax(-dist, dim=-1).mean(0)  # (K,) mean soft assignment over batch
        return -(avg_probs * avg_probs.clamp(min=eps).log()).sum()

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.codebook[indices], dim=-1)
