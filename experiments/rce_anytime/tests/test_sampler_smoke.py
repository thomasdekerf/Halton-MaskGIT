from __future__ import annotations

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
from experiments.rce_anytime.codecs import ToyKMeansCodec
from experiments.rce_anytime.sampling import rce_decode
from experiments.rce_anytime.train_rce import TinyTransformer


def test_sampler_smoke(tmp_path):
    dataset = datasets.CIFAR10(root=tmp_path, train=True, download=True, transform=transforms.ToTensor())
    subset = Subset(dataset, list(range(256)))
    codec = ToyKMeansCodec(patch_size=4)
    codec.fit_codebook(subset, num_codes=256, patch_size=4, max_iters=5, sample_patches=20_000)
    H = W = 8
    vocab = codec.vocab_size + 1
    model = TinyTransformer(vocab, H * W, dim=64, layers=2, heads=4)
    out = rce_decode(model, codec, H=H, W=W, T=4, commit_thresh=0.0, remask_thresh=0.0)
    masked = [int((t == codec.mask_token_id).sum()) for t in out["tokens_t"]]
    assert all(masked[i] >= masked[i + 1] for i in range(len(masked) - 1))
    assert len(out["images_t"]) == 4
    assert out["uncertainty_t"][0].shape == (1, H, W)
