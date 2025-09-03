"""Minimal test for positional and thumbnail encoders."""

import torch

from Network.positional import AbsXY
from Network.thumb_encoder import ThumbEncoder


def main() -> None:
    thumb = torch.randn(1, 3, 256, 256)
    enc = ThumbEncoder(out_dim=256)
    tokens = enc(thumb)
    print("context tokens shape:", tokens.shape)

    pos = AbsXY(max_size=512, dim=256)
    xy = torch.tensor([[10, 20]])
    emb = pos(xy)
    print("positional embedding shape:", emb.shape)


if __name__ == "__main__":
    main()

