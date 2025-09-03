# WSI Sparse Inpainting

This document describes the minimal steps to reproduce sparse inpainting on
Whole-Slide Images (WSIs) using the extensions added in this branch.

## Installation

```bash
conda env create -f env.yaml
conda activate maskgit
```

## Dataset preparation

1. Create a CSV file with the columns `slide_id,wsi_path,thumb_path`.
2. Extract VQ-VAE tokens for each slide and store them as `tokens/{slide_id}.npy`.

## Quick start

```bash
# Encode VQ tokens (example script, not provided)
python extract_vq_features.py --input slides.csv --output tokens/

# Launch training
python main.py --config Config/base_wsi_inpaint.yaml --train

# Reconstruct a slide
python tools/reconstruct_wsi.py --slide_csv slides.csv --vq_tokens_root tokens/ --out_dir preview/
```

Training progress can be monitored via TensorBoard.  Expect the cross-entropy
loss to decrease together with FID while token accuracy increases.  Preview
images stored under `results/wsi_inpaint/` should show progressively improved
reconstructions.

## Foundation model conditioning

To enable additional conditioning using embeddings from a pathology foundation
model, supply pre-computed embeddings in the data loader and pass the
`--use_foundation` flag when creating the trainer.

