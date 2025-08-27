"""RCE decoding loop implementing reversible anytime sampling."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from .schedulers import (
    cosine_reveal_schedule,
    confidence_topk_select,
    poisson_disk_select,
)


@torch.no_grad()
def rce_decode(
    transformer: nn.Module,
    codec,
    H: int,
    W: int,
    T: int = 8,
    k_schedule: List[float] | None = None,
    temp_schedule: List[float] | None = None,
    commit_thresh: float = 0.65,
    remask_thresh: float = 0.40,
    beam_k: int = 2,
    selector: str = "confidence",
    cond: Optional[Dict] = None,
) -> Dict[str, List[torch.Tensor]]:
    """Anytime decoding with reversible commitments.

    Args:
        transformer: Token model returning logits ``[B, N, V]``.
        codec: Token codec providing ``decode`` and ``mask_token_id``.
        H, W: Spatial resolution of the token grid.
        T: Number of decoding iterations.
        k_schedule: Fractions of tokens to reveal each iteration. If ``None`` a cosine
            schedule is used.
        temp_schedule: Optional temperature schedule for logits.
        commit_thresh: Confidence threshold above which tokens become committed.
        remask_thresh: If a committed token falls below this confidence it is re-masked.
        beam_k: Number of candidate tokens considered per position.
        selector: ``"confidence"`` or ``"poisson"`` for token selection.
        cond: Optional conditioning dict passed to the transformer (unused here).

    Returns
    -------
    dict
        ``{"tokens_t", "images_t", "uncertainty_t"}`` where each entry is a list over
        ``T`` iterations.
    """

    device = next(transformer.parameters()).device
    B = 1  # batching can be extended if needed
    N = H * W
    tokens = torch.full((B, N), codec.mask_token_id, dtype=torch.long, device=device)
    committed = torch.zeros((B, N), dtype=torch.bool, device=device)

    if k_schedule is None:
        k_schedule = cosine_reveal_schedule(T)
    if temp_schedule is None:
        temp_schedule = [1.0 for _ in range(T)]

    tokens_t: List[torch.Tensor] = []
    images_t: List[torch.Tensor] = []
    uncertainty_t: List[torch.Tensor] = []

    for t in range(T):
        logits = transformer(tokens)
        if logits.size(-1) > codec.vocab_size:
            logits[..., codec.vocab_size] = float("-inf")
        logits_vocab = logits[..., : codec.vocab_size]
        probs = torch.softmax(logits_vocab / temp_schedule[t], dim=-1)
        conf, best = probs.max(dim=-1)

        # re-mask low-confidence tokens
        low = (conf < remask_thresh) & committed
        committed[low] = False
        tokens[low] = codec.mask_token_id

        mask = ~committed
        k = max(1, int(k_schedule[t] * N))
        k = min(k, mask.sum().item())

        if k > 0:
            if selector == "poisson":
                conf_masked = conf[0, mask[0]]
                pool = torch.nonzero(mask[0], as_tuple=False).squeeze(1)
                rel = poisson_disk_select(conf_masked, k, H, W)
                sel = pool[rel].unsqueeze(0)
            else:
                sel = confidence_topk_select(conf, mask, k)
            for b in range(B):
                idx = sel[b]
                logits_b = logits_vocab[b, idx]
                probs_b = probs[b, idx]
                conf_b = conf[b, idx]
                topk_prob, topk_tok = probs_b.topk(beam_k, dim=-1)
                choose = topk_tok[:, 0]
                commit = conf_b >= commit_thresh
                tokens[b, idx[commit]] = choose[commit]
                committed[b, idx[commit]] = True

        tmp_tokens = tokens.clone()
        tmp_tokens[~committed] = best[~committed]
        imgs = codec.decode(tmp_tokens.view(B, H, W))
        tokens_t.append(tokens.view(B, H, W).clone())
        images_t.append(imgs)
        uncertainty_t.append((1 - conf).view(B, H, W))

    return {"tokens_t": tokens_t, "images_t": images_t, "uncertainty_t": uncertainty_t}

__all__ = ["rce_decode"]
