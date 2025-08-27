"""Reversible coupling block used for RCE.

The block implements an additive/affine coupling layer similar to those used in
normalizing flows. It allows information to be recovered so that already sampled tokens
can be reverted when their confidence decreases during anytime decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass
class MLP(nn.Module):
    """Small MLP with GELU activation."""

    in_dim: int
    hidden_dim: int
    out_dim: int

    def __post_init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RevCoupling(nn.Module):
    """Additive/affine reversible coupling.

    Args:
        dim: Feature dimension of the input ``z``.
        hidden_dim: Hidden size of the internal MLPs.
        mode: ``"additive"`` or ``"affine"``. In ``"additive"`` mode the coupling
            reduces to ``z1' = z1 + h(z2)``.

    Notes
    -----
    Given an input ``z = [z1, z2]`` split along the channel dimension, the forward
    transformation is::

        z2' = z2 + f(z1, y)
        z1' = z1 * exp(g(z2', y)) + h(z2', y)

    where ``y`` is an optional conditioning embedding. The inverse applies the reverse
    operations, ensuring exact invertibility.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        mode: Literal["additive", "affine"] = "affine",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.f = MLP(dim // 2, hidden_dim, dim // 2)
        self.g = MLP(dim // 2, hidden_dim, dim // 2)
        self.h = MLP(dim // 2, hidden_dim, dim // 2)

    def forward(self, z: torch.Tensor, y_embed: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the forward coupling transformation.

        Args:
            z: Input tensor of shape ``[B, N, C]``.
            y_embed: Optional conditioning tensor broadcastable to ``[B, N, C/2]``.

        Returns
        -------
        torch.Tensor
            Transformed tensor of the same shape as ``z``.
        """

        z1, z2 = z.chunk(2, dim=-1)
        if y_embed is not None:
            y1, y2 = y_embed.chunk(2, dim=-1)
        else:
            y1 = y2 = 0.0

        z2_ = z2 + self.f(z1 + y1)
        if self.mode == "affine":
            s = torch.clamp(self.g(z2_ + y2), -5.0, 5.0)
            z1_ = z1 * torch.exp(s) + self.h(z2_ + y2)
        else:
            z1_ = z1 + self.h(z2_ + y2)
        return torch.cat([z1_, z2_], dim=-1)

    def inverse(self, z: torch.Tensor, y_embed: torch.Tensor | None = None) -> torch.Tensor:
        """Inverse transformation of :meth:`forward`.

        Args:
            z: Output tensor of ``forward`` with shape ``[B, N, C]``.
            y_embed: Optional conditioning tensor broadcastable to ``[B, N, C/2]``.

        Returns
        -------
        torch.Tensor
            Recovered input tensor of the same shape.
        """

        z1_, z2_ = z.chunk(2, dim=-1)
        if y_embed is not None:
            y1, y2 = y_embed.chunk(2, dim=-1)
        else:
            y1 = y2 = 0.0

        if self.mode == "affine":
            s = torch.clamp(self.g(z2_ + y2), -5.0, 5.0)
            z1 = (z1_ - self.h(z2_ + y2)) * torch.exp(-s)
        else:
            z1 = z1_ - self.h(z2_ + y2)
        z2 = z2_ - self.f(z1 + y1)
        return torch.cat([z1, z2], dim=-1)

__all__ = ["RevCoupling"]
