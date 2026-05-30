"""EMAEncoder: exponential moving average wrapper for any nn.Module."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


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
