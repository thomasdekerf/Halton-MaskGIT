from __future__ import annotations

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.nn.functional as F

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
from experiments.rce_anytime.codecs import ToyKMeansCodec


def test_codec_smoke(tmp_path):
    dataset = datasets.CIFAR10(root=tmp_path, train=True, download=True, transform=transforms.ToTensor())
    subset = Subset(dataset, list(range(128)))
    codec = ToyKMeansCodec(patch_size=4)
    codec.fit_codebook(subset, num_codes=256, patch_size=4, max_iters=5, sample_patches=10_000)
    imgs = torch.stack([subset[i][0] for i in range(16)])
    tokens = codec.encode(imgs)
    recon = codec.decode(tokens)
    assert recon.shape == imgs.shape
    mse = F.mse_loss(recon, imgs).item()
    assert mse < 0.04
