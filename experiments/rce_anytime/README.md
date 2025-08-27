# RCE Anytime Experiment

This package implements a toy **Reversible + Anytime decoding (RCE)** experiment on top of the Halton-MaskGIT codebase. RCE augments a bidirectional token transformer with reversible coupling layers so that already generated tokens can be re-masked when their confidence decreases. Decoding proceeds in several iterations and can be interrupted at any time to obtain a valid image, providing an "anytime" generation property.

## Installation

The experiment only depends on a small set of libraries:

```bash
pip install torch torchvision numpy tqdm einops
# optional for nicer training loops
pip install lightning
```

## Usage

Train a tiny model on CIFAR‑10:

```bash
python -m experiments.rce_anytime.train_rce --data.root ./data --epochs 1
```

Sample from a trained checkpoint:

```bash
python -m experiments.rce_anytime.sample_rce --ckpt path/to.ckpt --num_samples 4
```

## Artifacts

Training produces checkpoints, TensorBoard curves and optional sample grids under `experiments/rce_anytime/.artifacts/`. Sampling creates per‑iteration image grids and a CSV file containing anytime metrics such as masked fraction and mean confidence.

