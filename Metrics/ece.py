"""Expected Calibration Error (ECE) metric."""

from __future__ import annotations

import torch


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    """Compute the Expected Calibration Error for classification probabilities."""

    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            ece += (upper - lower) * (acc - conf).abs()
    return ece


__all__ = ["expected_calibration_error"]

