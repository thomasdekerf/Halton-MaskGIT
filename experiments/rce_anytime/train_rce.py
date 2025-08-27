"""Command line interface for training the RCE experiment on CIFAR-10."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CIFARTokens
from .models import RCECore, RCEModule

try:
    import lightning as L
except Exception:
    L = None


class TinyTransformer(nn.Module):
    """Small bidirectional transformer used for the toy experiment."""

    def __init__(self, vocab: int, seq_len: int, dim: int = 512, layers: int = 12, heads: int = 8) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, dim))
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        self.to_logits = nn.Linear(dim, vocab)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:  # [B, N]
        x = self.token_emb(tokens) + self.pos_emb[:, : tokens.size(1)]
        x = self.transformer(x)
        return self.to_logits(x)


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RCE on CIFAR-10")
    parser.add_argument("--data.root", dest="data_root", default="./data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_codes", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--model.dim", dest="dim", type=int, default=512)
    parser.add_argument("--model.layers", dest="layers", type=int, default=4)
    parser.add_argument("--model.heads", dest="heads", type=int, default=8)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="32")
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    train_ds = CIFARTokens(args.data_root, train=True, num_codes=args.num_codes, patch_size=args.patch_size)
    val_ds = CIFARTokens(args.data_root, train=False, num_codes=args.num_codes, patch_size=args.patch_size)
    codec = train_ds.codec
    vocab = codec.vocab_size + 1  # include mask id
    seq_len = (32 // args.patch_size) ** 2
    model = TinyTransformer(vocab, seq_len, dim=args.dim, layers=args.layers, heads=args.heads)
    core = RCECore(model, codec, vocab_size=codec.vocab_size, mask_id=codec.mask_token_id)

    if L is None:
        optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        for epoch in range(args.epochs):
            for imgs, _ in loader:
                loss = core.training_loss(imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"epoch {epoch} train_ce={loss.item():.3f}")
        return

    module = RCEModule(core, lr=args.lr)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator="auto",
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":  # pragma: no cover
    main()
