"""Toy K-means image codec.

This module implements a very small vector quantisation scheme based on K-means. It
serves as a drop-in replacement for real VQ-VAE tokenisers so that the RCE experiment
can run without external weights. The codec operates on non-overlapping image patches
and represents each patch by the index of its nearest centroid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class ToyKMeansCodec:
    """Simple K-means based image codec.

    The codec splits an image into non-overlapping ``patch_size`` patches, flattens the
    RGB values and performs K-means clustering to obtain a codebook. Encoding assigns
    each patch to its nearest centroid and decoding reconstructs an image by placing the
    centroid colour back into the patch.

    Args:
        codebook: Optional initial codebook of shape ``[K, 3 * patch_size * patch_size]``.
        patch_size: Patch edge length in pixels.
    """

    codebook: Optional[torch.Tensor] = None
    patch_size: int = 4

    def fit_codebook(
        self,
        dataset: Dataset,
        num_codes: int = 512,
        patch_size: int | None = None,
        max_iters: int = 50,
        sample_patches: int = 100_000,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        """Fit the K-means codebook on the given dataset.

        Args:
            dataset: Dataset yielding images in ``[0, 1]`` of shape ``[C, H, W]``.
            num_codes: Number of K-means centroids.
            patch_size: Patch size; overrides the instance attribute when provided.
            max_iters: Maximum number of Lloyd iterations.
            sample_patches: Random subset of patches used for clustering.
            batch_size: Mini-batch size for data loading.
            device: Device used for clustering.

        Notes
        -----
        The implementation uses a simple Lloyd's algorithm with K-means++ initialisation
        and operates fully in PyTorch for simplicity. It is intended for small toy
        experiments and is not optimised for speed.
        """

        if patch_size is not None:
            self.patch_size = patch_size

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        patches = []
        for imgs, *_ in loader:
            # imgs: [B, C, H, W]
            B, C, H, W = imgs.shape
            imgs = imgs.to(device)
            patches.append(
                imgs.unfold(2, self.patch_size, self.patch_size)
                .unfold(3, self.patch_size, self.patch_size)
                .contiguous()
                .view(B, C, -1, self.patch_size, self.patch_size)
                .permute(0, 2, 1, 3, 4)
                .reshape(-1, C * self.patch_size * self.patch_size)
            )
            if sum(p.shape[0] for p in patches) >= sample_patches:
                break
        patches = torch.cat(patches, dim=0)[:sample_patches]

        # K-means++ initialisation
        inds = torch.randint(0, patches.size(0), (1,), device=device)
        centroids = patches[inds].clone()
        for _ in range(1, num_codes):
            dist = torch.cdist(patches, centroids, p=2) ** 2
            probs = dist.min(dim=1).values
            next_idx = torch.multinomial(probs, 1)
            centroids = torch.cat([centroids, patches[next_idx]], dim=0)

        for _ in range(max_iters):
            # Assign
            dist = torch.cdist(patches, centroids, p=2)
            assign = dist.argmin(dim=1)
            # Update
            for k in range(num_codes):
                mask = assign == k
                if mask.any():
                    centroids[k] = patches[mask].mean(dim=0)

        self.codebook = centroids.detach().to(device)

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Number of codes in the codebook."""

        assert self.codebook is not None, "Codebook not initialised"
        return self.codebook.size(0)

    # ------------------------------------------------------------------
    @property
    def mask_token_id(self) -> int:
        """ID used to denote a masked patch."""

        return self.vocab_size  # reserve last index

    # ------------------------------------------------------------------
    def encode(self, images: torch.Tensor) -> torch.LongTensor:
        """Encode images into token grids.

        Args:
            images: Tensor of shape ``[B, C, H, W]`` with values in ``[0, 1]``.

        Returns
        -------
        torch.LongTensor
            Token indices of shape ``[B, H / p, W / p]`` where ``p`` is ``patch_size``.
        """

        assert self.codebook is not None, "Codebook not initialised"
        B, C, H, W = images.shape
        p = self.patch_size
        patches = (
            images.unfold(2, p, p)
            .unfold(3, p, p)
            .contiguous()
            .view(B, C, -1, p, p)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, C * p * p)
        )
        dist = torch.cdist(patches, self.codebook, p=2)
        tokens = dist.argmin(dim=1).view(B, H // p, W // p)
        return tokens

    # ------------------------------------------------------------------
    def decode(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Decode tokens back into images.

        Args:
            tokens: Tensor of shape ``[B, H, W]`` with code indices.

        Returns
        -------
        torch.Tensor
            Reconstructed images of shape ``[B, 3, H * p, W * p]``.
        """

        assert self.codebook is not None, "Codebook not initialised"
        B, H, W = tokens.shape
        p = self.patch_size
        centroids = self.codebook[tokens.view(-1)].view(B, H, W, 3, p, p)
        imgs = centroids.permute(0, 3, 1, 4, 2, 5).contiguous()
        imgs = imgs.view(B, 3, H * p, W * p)
        return imgs.clamp(0.0, 1.0)

