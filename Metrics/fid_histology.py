"""FID computation tailored for histology patches."""

from __future__ import annotations

from pathlib import Path

from cleanfid import fid


def compute_fid(real_dir: str | Path, fake_dir: str | Path) -> float:
    """Compute Fréchet Inception Distance between two directories of images."""

    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)
    real_paths = sorted(real_dir.glob("*.png"))
    fake_paths = sorted(fake_dir.glob("*.png"))
    if not real_paths or not fake_paths:
        raise FileNotFoundError("No images found for FID computation")
    return float(fid.compute_fid(real_dir, fake_dir))


__all__ = ["compute_fid"]

