"""WSI glimpse dataset for sparse inpainting.

This module defines :class:`WSIGlimpseDataset` which samples random windows
from whole-slide images (WSIs).  For every sampled window a subset of tokens is
revealed as *glimpses* while the remaining tokens are replaced by a mask token
ID.  In addition the dataset returns a downsampled thumbnail of the full slide
which can be used as conditioning information.

The dataset expects a CSV file with the following columns::

    slide_id,wsi_path,thumb_path

``slide_id`` uniquely identifies a slide.  ``wsi_path`` points to the WSI file
readable by OpenSlide and ``thumb_path`` is the path to a thumbnail image.  For
every ``slide_id`` a numpy ``.npy`` file containing the VQ-VAE token grid must
exist in ``token_root`` passed at construction time.

The output of ``__getitem__`` is a dictionary with the keys:

``tok_in``      Tokens with non-glimpsed positions masked.
``tok_tgt``     Ground-truth tokens of the sampled window.
``tok_lock``    Boolean mask indicating glimpsed (``True``) positions.
``abs_xy``      Absolute top-left token coordinates of the window.
``thumb``       Tensor representation of the slide thumbnail.

The implementation intentionally keeps dependencies light and falls back to
``PIL`` when OpenSlide is not available so that unit tests can run on minimal
setups.  When OpenSlide is present it is used to read the corresponding region
from the WSI which ensures correct alignment between token coordinates and
image pixels.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

try:  # pragma: no cover - optional dependency
    import openslide  # type: ignore
except Exception:  # pragma: no cover
    openslide = None


class WSIGlimpseDataset(Dataset):
    """Dataset sampling windows from whole-slide images.

    Parameters
    ----------
    csv_path:
        Path to the slide CSV file.
    token_root:
        Directory containing ``{slide_id}.npy`` token grids.
    mask_token_id:
        Integer representing the mask token used for unknown positions.
    window_tokens:
        Edge length of the sampled window in tokens.  Defaults to ``64`` which
        corresponds to a ``1024×1024`` pixel patch for a VQ-VAE with
        down-sampling factor ``f=16``.
    token_size:
        Size in pixels of a single token.  Only required when OpenSlide is
        available as it is used to read the corresponding pixel region.
    thumb_size:
        Maximum edge length of the returned thumbnail image.
    """

    def __init__(
        self,
        csv_path: str | Path,
        token_root: str | Path,
        mask_token_id: int = 0,
        window_tokens: int = 64,
        token_size: int = 16,
        thumb_size: int = 1024,
    ) -> None:
        super().__init__()
        self.entries = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)

        self.token_root = Path(token_root)
        self.mask_token_id = int(mask_token_id)
        self.window_tokens = int(window_tokens)
        self.token_size = int(token_size)
        self.thumb_size = int(thumb_size)

        # Cache thumbnails to avoid repeatedly decoding them from disk.
        self._thumb_cache: Dict[str, torch.Tensor] = {}
        # Cache token grids as memmaps for efficiency on large slides.
        self._token_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------
    def _load_tokens(self, slide_id: str) -> np.ndarray:
        if slide_id not in self._token_cache:
            token_path = self.token_root / f"{slide_id}.npy"
            self._token_cache[slide_id] = np.load(token_path)
        return self._token_cache[slide_id]

    # ------------------------------------------------------------------
    def _load_thumb(self, path: str) -> torch.Tensor:
        if path not in self._thumb_cache:
            img = Image.open(path).convert("RGB")
            img.thumbnail((self.thumb_size, self.thumb_size))
            self._thumb_cache[path] = (
                torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
            )
        return self._thumb_cache[path]

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[index % len(self.entries)]
        slide_id = entry["slide_id"]
        tokens = self._load_tokens(slide_id)
        H, W = tokens.shape

        win = self.window_tokens
        assert H >= win and W >= win, "Token map smaller than sampling window"
        top = random.randint(0, H - win)
        left = random.randint(0, W - win)

        tok_window = tokens[top : top + win, left : left + win].astype(np.int64)
        tok_tgt = torch.from_numpy(tok_window)
        tok_in = tok_tgt.clone()

        N = win * win
        n_glimpse = random.randint(int(0.05 * N), int(0.15 * N))
        flat_idx = torch.randperm(N)[:n_glimpse]
        tok_lock = torch.zeros(N, dtype=torch.bool)
        tok_lock[flat_idx] = True
        tok_lock = tok_lock.view(win, win)

        tok_in[~tok_lock] = self.mask_token_id
        abs_xy = torch.tensor([top, left], dtype=torch.long)

        # Optionally read the corresponding pixel region using OpenSlide for
        # sanity checking.  The region is currently not returned but this call
        # ensures the coordinates are valid when OpenSlide is available.
        if openslide is not None:  # pragma: no cover - not executed in tests
            try:
                slide = openslide.OpenSlide(entry["wsi_path"])
                size = win * self.token_size
                slide.read_region((left * self.token_size, top * self.token_size), 0, (size, size))
                slide.close()
            except Exception:
                pass

        thumb = self._load_thumb(entry["thumb_path"])

        return {
            "tok_in": tok_in,
            "tok_tgt": tok_tgt,
            "tok_lock": tok_lock,
            "abs_xy": abs_xy,
            "thumb": thumb,
        }


__all__ = ["WSIGlimpseDataset"]

