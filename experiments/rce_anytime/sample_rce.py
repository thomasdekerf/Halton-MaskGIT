"""Sampling script for the RCE experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image

from .data import CIFARTokens
from .models import RCECore
from .train_rce import TinyTransformer
from .sampling import rce_decode


def parse_args(argv: Any = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from an RCE checkpoint")
    p.add_argument("--ckpt", type=str, required=False, help="Path to model state dict")
    p.add_argument("--out_dir", type=str, default="./samples")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--selector", type=str, default="confidence")
    p.add_argument("--commit_thresh", type=float, default=0.65)
    p.add_argument("--remask_thresh", type=float, default=0.40)
    p.add_argument("--data.root", dest="data_root", default="./data")
    p.add_argument("--num_codes", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--model.dim", dest="dim", type=int, default=512)
    p.add_argument("--model.layers", dest="layers", type=int, default=4)
    p.add_argument("--model.heads", dest="heads", type=int, default=8)
    return p.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    ds = CIFARTokens(args.data_root, train=False, num_codes=args.num_codes, patch_size=args.patch_size)
    codec = ds.codec
    vocab = codec.vocab_size + 1
    H = W = 32 // args.patch_size
    seq_len = H * W
    model = TinyTransformer(vocab, seq_len, dim=args.dim, layers=args.layers, heads=args.heads)
    core = RCECore(model, codec, vocab_size=codec.vocab_size, mask_id=codec.mask_token_id)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        core.load_state_dict(state)
    outs = rce_decode(model, codec, H=H, W=W, T=args.T, selector=args.selector, commit_thresh=args.commit_thresh, remask_thresh=args.remask_thresh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, img in enumerate(outs["images_t"]):
        save_image(img, out_dir / f"grid_iter_{t}.png")


if __name__ == "__main__":  # pragma: no cover
    main()
