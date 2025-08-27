"""Scheduling utilities for RCE sampling."""

from __future__ import annotations

import math
from typing import List

import torch


def cosine_reveal_schedule(T: int) -> List[float]:
    """Cosine annealed reveal schedule.

    Produces ``T`` fractions summing to one such that early iterations reveal more
    tokens. The schedule follows a cosine decay.
    """

    vals = [0.5 * (1 - math.cos(math.pi * (i + 1) / T)) for i in range(T)]
    vals = vals[::-1]
    s = sum(vals)
    return [v / s for v in vals]


def confidence_topk_select(conf: torch.Tensor, mask: torch.BoolTensor, k: int) -> torch.LongTensor:
    """Select top ``k`` positions by confidence.

    Args:
        conf: Confidence scores of shape ``[B, N]``.
        mask: Boolean mask with the same shape where ``True`` denotes masked tokens.
        k: Number of indices to return per batch.
    """

    masked_conf = conf.masked_fill(~mask, float("-inf"))
    idx = torch.topk(masked_conf, k, dim=1).indices
    return idx


def poisson_disk_select(
    masked_conf: torch.Tensor,
    k: int,
    H: int,
    W: int,
    min_dist: int = 2,
) -> torch.LongTensor:
    """Select ``k`` positions using Poisson disk sampling.

    This is a simple ``O(kN)`` implementation suitable for small ``H`` and ``W``.

    Args:
        masked_conf: Tensor of shape ``[N]`` with confidence values for masked tokens.
        k: Number of points to select.
        H: Height of the token grid.
        W: Width of the token grid.
        min_dist: Minimum distance (in tokens) between selected points.
    """

    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1).view(-1, 2)
    conf = masked_conf.clone()
    idx_sorted = torch.argsort(conf, descending=True)
    selected = []
    for idx in idx_sorted.tolist():
        if len(selected) >= k:
            break
        y, x = coords[idx]
        if all((coords[s] - torch.tensor([y, x])).pow(2).sum() >= min_dist**2 for s in selected):
            selected.append(idx)
    return torch.tensor(selected, dtype=torch.long)

__all__ = [
    "cosine_reveal_schedule",
    "poisson_disk_select",
    "confidence_topk_select",
]
