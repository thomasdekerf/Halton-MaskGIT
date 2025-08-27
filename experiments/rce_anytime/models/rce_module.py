"""RCE model wrappers.

Combines a transformer with a codec and optionally exposes a LightningModule for
training.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import nn
import torch.nn.functional as F

try:  # optional dependency
    import lightning as L
except Exception:  # pragma: no cover
    L = None  # type: ignore


@dataclass
class RCECore(nn.Module):
    """Core model combining transformer and codec."""

    transformer: nn.Module
    codec: object
    vocab_size: int
    mask_id: int

    def __post_init__(self) -> None:
        super().__init__()

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Forward pass through the transformer."""

        return self.transformer(tokens)

    def training_loss(self, images: torch.Tensor) -> torch.Tensor:
        """Compute masked token prediction loss."""

        tokens = self.codec.encode(images)
        B, H, W = tokens.shape
        tokens = tokens.view(B, -1)
        mask_ratio = random.uniform(0.3, 0.6)
        mask = torch.rand_like(tokens.float()) < mask_ratio
        inputs = tokens.clone()
        inputs[mask] = self.mask_id
        logits = self.forward(inputs)
        return F.cross_entropy(logits[mask], tokens[mask], reduction="mean")


if L is not None:  # pragma: no cover - optional lightning wrapper
    class RCEModule(L.LightningModule):
        """Lightning wrapper around :class:`RCECore`."""

        def __init__(self, core: RCECore, lr: float = 2e-4, weight_decay: float = 0.01):
            super().__init__()
            self.core = core
            self.lr = lr
            self.weight_decay = weight_decay

        def training_step(self, batch, batch_idx):
            imgs, _ = batch
            loss = self.core.training_loss(imgs)
            self.log("train/ce", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.core.parameters(), lr=self.lr, weight_decay=self.weight_decay)
else:  # simple placeholder when Lightning is unavailable
    class RCEModule:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - placeholder
            raise RuntimeError("lightning is required for RCEModule")

__all__ = ["RCECore", "RCEModule"]
