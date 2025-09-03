"""WSI reconstruction and active acquisition utilities."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


def _load_slide_csv(csv_path: Path) -> List[dict]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute token-wise entropy from logits."""

    probs = torch.softmax(logits, dim=-1)
    return -(probs * (probs + 1e-8).log()).sum(dim=-1)


def pick_glimpses(entropy: torch.Tensor, k: int) -> List[Tuple[int, int]]:
    """Select the ``k`` most uncertain token locations with simple NMS.

    The non-maximum suppression radius is one token which is sufficient for the
    low-resolution token grid used here.
    """

    ent = entropy.clone().view(-1)
    coords = []
    for _ in range(k):
        idx = int(ent.argmax())
        y = idx // entropy.size(1)
        x = idx % entropy.size(1)
        coords.append((y, x))
        # NMS radius = 1 token
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < entropy.size(0) and 0 <= nx < entropy.size(1):
                    ent[ny * entropy.size(1) + nx] = -1
    return coords


def reconstruct_wsi(args: argparse.Namespace) -> None:
    """Placeholder WSI reconstruction pipeline.

    The function demonstrates the high-level steps required for sparse
    inpainting with uncertainty estimation but does not perform real
    inpainting since no model is provided.  Instead it averages the known token
    values and saves the resulting low-resolution token grid as an image.
    """

    slides = _load_slide_csv(Path(args.slide_csv))
    token_root = Path(args.vq_tokens_root)

    for slide in slides:
        slide_id = slide["slide_id"]
        tokens = np.load(token_root / f"{slide_id}.npy")
        tokens = torch.from_numpy(tokens)

        # Placeholder reconstruction: simply normalise tokens.
        recon = tokens.float() / tokens.max()
        img = (recon.numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(Path(args.out_dir) / f"{slide_id}_preview.tif")

        # Fake entropy map for demonstration.
        logits = torch.randn(tokens.shape[0], tokens.shape[1], args.vocab_size)
        ent = _entropy(logits)
        ent_img = (ent / ent.max()).numpy()
        Image.fromarray((ent_img * 255).astype(np.uint8)).save(
            Path(args.out_dir) / f"{slide_id}_entropy.png"
        )

        if args.next_k > 0:
            coords = pick_glimpses(ent, args.next_k)
            print(f"Next glimpses for {slide_id}: {coords}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WSI reconstruction")
    p.add_argument("--slide_csv", type=str, required=True)
    p.add_argument("--vq_tokens_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="results/wsi_inpaint")
    p.add_argument("--vocab_size", type=int, default=1024)
    p.add_argument("--next_k", type=int, default=0, help="pick top-K uncertain glimpses")
    return p


if __name__ == "__main__":
    parser = build_parser()
    reconstruct_wsi(parser.parse_args())

