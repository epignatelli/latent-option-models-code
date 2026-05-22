"""Core building blocks: Spatio-Temporal Transformer and Vector Quantizer.

Architecture mirrors the latent-molecule-generation codebase, extended to
handle spatiotemporal observation sequences (T, H, W) via factored
spatial + temporal attention (TimeSformer-style).
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------------------------------------------------------------------- #
# --- Base ------------------------------------------------------------------ #
# --------------------------------------------------------------------------- #


class SerialisableModule(nn.Module):
    """nn.Module with save / load helpers."""

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str) -> None:
        config = {k: v for k, v in vars(self).items() if not isinstance(v, nn.Module)}
        torch.save(
            {"config": config, "params": self.state_dict(), "class": self.__class__.__name__}, path
        )
        logging.info("Saved %s to %s", self.__class__.__name__, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> SerialisableModule:
        dic = torch.load(path, map_location=device, weights_only=False)
        if dic["class"] != cls.__name__:
            raise ValueError(f"Checkpoint is {dic['class']}, loading as {cls.__name__}")
        sig = inspect.signature(cls.__init__).parameters
        cfg = {k: v for k, v in dic["config"].items() if k in sig}
        obj = cls(**cfg)
        obj.load_state_dict(dic["params"])
        logging.info("Loaded %s from %s", cls.__name__, path)
        return obj


# --------------------------------------------------------------------------- #
# --- Primitives ------------------------------------------------------------ #
# --------------------------------------------------------------------------- #


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
    """Multi-head self-attention with optional causal masking.

    Supports Flash Attention (PyTorch ≥ 2.0) with graceful fallback.
    Custom additive masks (e.g. option-token mask) are passed via attn_mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.causal = causal
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        head_dim = C // self.n_heads
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal and attn_mask is None,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            if self.causal:
                mask_val = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                att = att.masked_fill(mask_val, float("-inf"))
            if attn_mask is not None:
                att = att + attn_mask
            att = self.attn_drop(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


# --------------------------------------------------------------------------- #
# --- Spatio-Temporal Transformer ------------------------------------------- #
# --------------------------------------------------------------------------- #


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
        self.spatial_attn = SelfAttention(d_model, n_heads, dropout, bias, causal=False)
        self.temporal_attn = SelfAttention(d_model, n_heads, dropout, bias, causal=causal_temporal)
        self.mlp = MLP(d_model, dropout, bias)

    def forward(self, x: torch.Tensor, temporal_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, S, D)
            temporal_mask: additive mask of shape (T, T) or (1, 1, T, T)
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
        xt = xt + self.temporal_attn(self.ln_t(xt), attn_mask=temporal_mask)
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

    def forward(self, x: torch.Tensor, temporal_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, S, D) — already char-embedded, without positional info
            temporal_mask: optional additive mask (T, T)
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


# --------------------------------------------------------------------------- #
# --- Screen tokenisation --------------------------------------------------- #
# --------------------------------------------------------------------------- #

CHAR_VOCAB  = 256
COLOR_VOCAB = 32
TOKEN_VOCAB = CHAR_VOCAB * COLOR_VOCAB  # 8192


def tokenise(x: torch.Tensor) -> torch.Tensor:
    """Map (..., H, W, 2) char+color pairs to (..., H, W) integer token IDs.

    token_id = char * COLOR_VOCAB + color  ∈ [0, TOKEN_VOCAB)
    """
    return x[..., 0].long() * COLOR_VOCAB + x[..., 1].long()


class ScreenTokeniser(nn.Module):
    """Stateless module wrapping tokenise() so it composes with nn.Sequential
    and survives torch.compile.

    Input:  (..., H, W, 2) — last dim is (char uint8, color uint8)
    Output: (..., H, W)    — long token IDs in [0, TOKEN_VOCAB)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tokenise(x)


# --------------------------------------------------------------------------- #
# --- Patch Embedding ------------------------------------------------------- #
# --------------------------------------------------------------------------- #


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
        assert obs_h % patch_size == 0 and obs_w % patch_size == 0, (
            f"obs_h={obs_h} and obs_w={obs_w} must both be divisible by patch_size={patch_size}"
        )
        self.patch_size = patch_size
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.d_model = d_model
        self.n_tokens = (obs_h // patch_size) * (obs_w // patch_size)

        self.char_embed = nn.Embedding(vocab_size, d_model)
        self.patch_proj = (
            nn.Linear(patch_size ** 2 * d_model, d_model, bias=bias)
            if patch_size > 1 else None
        )

        self.register_buffer("token_usage", torch.zeros(vocab_size, dtype=torch.long))

        def _usage_hook(_module, inputs, _output):
            ids = inputs[0].detach().reshape(-1)
            self.token_usage.index_add_(
                0, ids, torch.ones(ids.numel(), dtype=torch.long, device=ids.device)
            )

        self.char_embed.register_forward_hook(_usage_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W) long — token IDs
        Returns:
            (B, T, n_tokens, d_model)
        """
        B, T, H, W = x.shape
        P, D = self.patch_size, self.d_model

        emb = self.char_embed(x)  # (B, T, H, W, D)

        if self.patch_proj is not None:
            emb = emb.reshape(B, T, H // P, P, W // P, P, D)
            emb = emb.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # (B, T, H/P, W/P, P, P, D)
            emb = emb.reshape(B, T, self.n_tokens, P * P * D)
            emb = self.patch_proj(emb)                             # (B, T, n_tokens, D)
        else:
            emb = emb.reshape(B, T, self.n_tokens, D)

        return emb


# --------------------------------------------------------------------------- #
# --- Vector Quantizer ------------------------------------------------------ #
# --------------------------------------------------------------------------- #


@dataclass
class VQConfig:
    latent_dim: int = 64
    num_options: int = 256
    dropout: float = 0.1
    entropy_weight: float = 0.01
    vq_beta: float = 0.25
    vq_reset_thresh: int = 100


class VectorQuantizer(nn.Module):
    """EMA-free VQ with cosine-distance assignment, straight-through estimator,
    entropy regularisation and optional dead-code reset.
    """

    def __init__(
        self,
        latent_dim: int,
        num_options: int,
        dropout: float,
        entropy_weight: float,
        vq_beta: float,
        vq_reset_thresh: int = 100,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_options = num_options
        self.entropy_weight = entropy_weight
        self.vq_beta = vq_beta
        self.vq_reset_thresh = vq_reset_thresh
        self.drop = nn.Dropout(dropout)

        bound = (3 / latent_dim) ** 0.5
        self.codebook = nn.Parameter(torch.empty(num_options, latent_dim).uniform_(-bound, bound))
        self._normalize_codebook()
        self.register_buffer("last_active", torch.zeros(num_options, dtype=torch.long))

    def _normalize_codebook(self) -> None:
        with torch.no_grad():
            self.codebook.data.copy_(F.normalize(self.codebook.data, dim=-1))

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
        z_hard = self.codebook[indices]

        # Dead-code reset
        if self.training and self.vq_reset_thresh > 0:
            self.last_active += 1
            self.last_active[indices.view(-1).unique()] = 0
            dead = (self.last_active >= self.vq_reset_thresh).nonzero(as_tuple=True)[0]
            if dead.numel():
                alive = (self.last_active < self.vq_reset_thresh).nonzero(as_tuple=True)[0]
                if alive.numel():
                    src = alive[torch.randint(alive.numel(), (dead.numel(),), device=z.device)]
                    with torch.no_grad():
                        self.codebook.data[dead] = self.codebook.data[src]
                        self.last_active[dead] = 0

        z_q = z + (z_hard - z).detach()  # straight-through

        commit_loss = F.mse_loss(z, z_hard.detach())
        q_loss = F.mse_loss(z_hard, z.detach())
        entropy = self._entropy(indices)
        vq_loss = q_loss + self.vq_beta * commit_loss - self.entropy_weight * entropy

        return (
            z_q,
            {
                "vq_loss": vq_loss,
                "q_loss": q_loss,
                "commit_loss": commit_loss,
                "entropy": entropy,
            },
            indices,
        )

    def _entropy(self, indices: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        counts = torch.bincount(indices.view(-1), minlength=self.num_options).float()
        p = counts / counts.sum()
        return -(p * p.clamp(min=eps).log()).sum()

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook[indices]
