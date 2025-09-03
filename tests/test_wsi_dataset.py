"""Quick smoke test for :mod:`Dataset.wsi_inpaint_dataset`.

The goal of this script is to exercise the ``WSIGlimpseDataset`` and ensure that
it returns the expected keys.  Real WSIs are not required; the test fabricates a
minimal dummy slide and token grid on the fly so that it can run in constrained
environments such as the evaluation container.

Running the script will iterate over five samples from the dataset, print the
shapes of the returned tensors and save a visualisation of the input and target
token grids to ``tests/``.
"""

import os
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from Dataset.wsi_inpaint_dataset import WSIGlimpseDataset


def _prepare_dummy(tmp: Path) -> Path:
    """Create a tiny dummy dataset for testing purposes."""

    tmp.mkdir(exist_ok=True, parents=True)

    # Create a dummy WSI/thumbnail image.
    wsi = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), np.uint8))
    wsi_path = tmp / "slide.tif"
    wsi.save(wsi_path)
    thumb_path = tmp / "thumb.png"
    wsi.resize((512, 512)).save(thumb_path)

    # Dummy token grid (128×128 tokens).
    tokens = np.random.randint(0, 1024, (128, 128), np.int64)
    token_root = tmp / "tokens"
    token_root.mkdir(exist_ok=True)
    np.save(token_root / "slide.npy", tokens)

    # CSV description.
    csv_path = tmp / "slides.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_id", "wsi_path", "thumb_path"])
        writer.writerow(["slide", str(wsi_path), str(thumb_path)])

    return csv_path, token_root


def main() -> None:
    root = Path("tests/_tmp")
    csv_path, token_root = _prepare_dummy(root)

    ds = WSIGlimpseDataset(csv_path, token_root, mask_token_id=1024)
    for i in range(5):
        sample = ds[i]
        print(
            f"Sample {i}: tok_in {sample['tok_in'].shape}, thumb {sample['thumb'].shape}, abs_xy {sample['abs_xy'].tolist()}"
        )

        # Visualise and save token grids for manual inspection.
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(sample["tok_in"].numpy())
        axs[0].set_title("tok_in")
        axs[1].imshow(sample["tok_tgt"].numpy())
        axs[1].set_title("tok_tgt")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(root / f"sample_{i}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()

