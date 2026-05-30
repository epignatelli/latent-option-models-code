"""Screen tokenisation: (H, W, 2) char+color pairs → integer token IDs."""

from __future__ import annotations

import torch
import torch.nn as nn

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
