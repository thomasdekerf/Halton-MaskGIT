"""Trainer for WSI sparse inpainting.

This is a lightweight adaptation of the repository's MaskGIT trainer.  It is
*not* a full reproduction of the research code but provides the scaffolding
required for experimentation and unit tests.  The trainer consumes tokenised
patches along with absolute positional encodings and thumbnail context tokens
and optimises a transformer model to inpaint the masked regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from Network.positional import AbsXY
from Network.thumb_encoder import ThumbEncoder


@dataclass
class InpaintConfig:
    vocab_size: int
    mask_token_id: int
    steps: int = 12
    use_thumbnail: bool = True
    xy_pos_dim: int = 256
    use_foundation: bool = False


class InpaintTrainer:
    """Simplified inpainting trainer.

    Parameters
    ----------
    model:
        Transformer model operating on token sequences.
    codec:
        VQ-VAE style codec providing ``decode`` for visualisation.
    config:
        Training configuration.
    thumb_encoder:
        Optional thumbnail encoder instance.  Used when ``use_thumbnail`` is
        ``True``.
    """

    def __init__(
        self,
        model: nn.Module,
        codec: nn.Module,
        config: InpaintConfig,
        thumb_encoder: Optional[ThumbEncoder] = None,
        use_foundation: bool | None = None,
    ) -> None:
        self.model = model
        self.codec = codec
        self.config = config
        self.thumb_encoder = thumb_encoder if config.use_thumbnail else None
        if use_foundation is not None:
            config.use_foundation = use_foundation
        self.xy_embed = AbsXY(dim=config.xy_pos_dim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.mask_token_id)

    # ------------------------------------------------------------------
    def training_step(self, batch: dict, step: int) -> torch.Tensor:
        """Perform a single optimisation step.

        The implementation is intentionally minimal and serves as a placeholder
        for a full-fledged MaskGIT training loop.  It demonstrates how absolute
        positional encodings and thumbnail context tokens could be integrated.
        """

        tok_in = batch["tok_in"].view(batch["tok_in"].size(0), -1)
        tok_tgt = batch["tok_tgt"].view(batch["tok_tgt"].size(0), -1)
        tok_lock = batch["tok_lock"].view(batch["tok_lock"].size(0), -1)

        # Position embeddings
        B, N = tok_in.shape
        coords = torch.stack(torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij"), dim=-1)
        coords = coords.view(1, N, 2).repeat(B, 1, 1).to(tok_in.device)
        pos = self.xy_embed(coords)

        # Thumbnail context
        context = None
        if self.thumb_encoder is not None and "thumb" in batch:
            context = self.thumb_encoder(batch["thumb"]).transpose(0, 1)

        if self.config.use_foundation and "foundation" in batch:
            foundation = batch["foundation"]
            context = foundation if context is None else torch.cat([context, foundation], dim=1)

        logits = self.model(tok_in, pos_embed=pos, context=context)
        loss = self.criterion(logits.view(-1, self.config.vocab_size), tok_tgt.view(-1))
        loss = (loss * (~tok_lock.view(-1))).mean()
        return loss


__all__ = ["InpaintTrainer", "InpaintConfig"]

