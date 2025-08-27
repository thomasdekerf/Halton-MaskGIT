"""CIFAR-10 dataset wrapper that yields images and toy codec tokens."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from ..codecs import ToyKMeansCodec


class CIFARTokens(Dataset):
    """CIFAR-10 images paired with tokens from :class:`ToyKMeansCodec`.

    On the first run a codebook is fitted and cached under ``.artifacts`` so that
    subsequent runs are fast.
    """

    def __init__(self, root: str, train: bool = True, num_codes: int = 512, patch_size: int = 4) -> None:
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transforms.ToTensor())
        art_dir = Path(__file__).resolve().parents[1] / ".artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        ckpt = art_dir / f"toy_codec_{num_codes}_{patch_size}.pt"
        self.codec = ToyKMeansCodec(patch_size=patch_size)
        if ckpt.exists():
            self.codec.codebook = torch.load(ckpt)
        else:
            self.codec.fit_codebook(self.dataset, num_codes=num_codes, patch_size=patch_size)
            torch.save(self.codec.codebook, ckpt)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.dataset[idx]
        tokens = self.codec.encode(img.unsqueeze(0))[0]
        return img, tokens

__all__ = ["CIFARTokens"]
