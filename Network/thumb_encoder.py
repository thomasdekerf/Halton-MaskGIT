"""Thumbnail encoder used for WSI conditioning."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class ThumbEncoder(nn.Module):
    """Encode slide thumbnails into a sequence of context tokens.

    The implementation is intentionally lightweight: a ResNet-18 backbone is
    used to extract a spatial feature map which is then projected to the desired
    dimensionality.  The feature map is flattened into a ``[B, N, D]`` token
    sequence suitable for cross-attention with the transformer model.
    """

    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(512, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        x:
            Input thumbnails of shape ``[B, 3, H, W]`` scaled to ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Context tokens of shape ``[B, N, out_dim]`` where ``N`` is the
            number of spatial positions of the ResNet feature map.
        """

        feat = self.features(x)
        feat = self.proj(feat)  # [B, D, H', W']
        B, D, H, W = feat.shape
        return feat.view(B, D, H * W).permute(0, 2, 1)


__all__ = ["ThumbEncoder"]

