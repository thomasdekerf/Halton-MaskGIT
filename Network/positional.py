"""Positional encodings for absolute XY coordinates."""

from __future__ import annotations

import torch
from torch import nn


class AbsXY(nn.Module):
    """Learned embedding for absolute ``(x, y)`` positions.

    The module maintains two embedding tables, one for the ``x`` coordinate and
    one for ``y``.  The resulting embeddings are concatenated to form a vector
    of dimension ``dim``.
    """

    def __init__(self, max_size: int = 65_536, dim: int = 256) -> None:
        """Parameters
        ----------
        max_size:
            Maximum coordinate value expected for ``x`` and ``y``.  The
            embeddings are defined in the range ``[0, max_size)``.
        dim:
            Dimensionality of the returned embedding.  Must be even because the
            ``x`` and ``y`` tables each use ``dim // 2`` dimensions.
        """

        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.x_embed = nn.Embedding(max_size, dim // 2)
        self.y_embed = nn.Embedding(max_size, dim // 2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Embed absolute coordinates.

        Parameters
        ----------
        coords:
            Tensor of shape ``[..., 2]`` containing ``(x, y)`` pairs.

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape ``[..., dim]``.
        """

        x = self.x_embed(coords[..., 0])
        y = self.y_embed(coords[..., 1])
        return torch.cat([x, y], dim=-1)


__all__ = ["AbsXY"]

