# OFA-Diffusion Compression

Code for **"OFA-Diffusion Compression: Compressing Diffusion Model in One-Shot Manner"**

---

## Overview

OFA-Diffusion trains a single network that supports multiple compression ratios simultaneously, eliminating the need to train a separate model for each target size. At inference time, any subnet can be extracted and optionally fine-tuned in a small number of additional steps.

We provide implementations for three backbones:

| Backbone | Dataset | Script |
|---|---|---|
| U-ViT (small) | CelebA 64×64 | `scripts/run_celeba64_ofa.sh` |
| EDM (VP, ddpmpp) | CIFAR-10 32×32 | `scripts/run_cifar10_ofa.sh` |
| Stable Diffusion v1.5 | MS-COCO 2014 | `scripts/run_coco_sd_ofa.sh` |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Pipeline

Each backbone follows the same four-step pipeline:

```
prune → train_ofa → extract → finetune → eval_fid
```

1. **Prune** — Compute Taylor importance scores and build per-P-value channel masks.
2. **Train OFA** — Joint training across all subnets (sandwich rule + weighted sampling).
3. **Extract** — Physically remove pruned weights to produce standalone subnet checkpoints.
4. **Fine-tune** — Short per-subnet fine-tuning to recover quality.
5. **Eval** — Generate samples and compute FID.

---

## Quick Start

```bash
# UViT — CelebA 64x64
GPUS=0,1,2,3 bash scripts/run_celeba64_ofa.sh

# EDM — CIFAR-10 32x32
GPUS=4 bash scripts/run_cifar10_ofa.sh

# Stable Diffusion v1.5 — MS-COCO
GPUS=4 bash scripts/run_coco_sd_ofa.sh
```

Set `OUTDIR=<path>` to control the output directory (default: `outputs/<task>_ofa`).

---

## Repository Structure

```
uvit/        U-ViT OFA pipeline  (prune / train_ofa / extract / finetune / eval_fid)
sd/          Stable Diffusion OFA pipeline
edm/         EDM OFA pipeline
third_party/ Third-party dependencies (NVlabs/edm)
scripts/     Example end-to-end run scripts
```

---

## Citation

```bibtex
@article{ofadiffusion2025,
  title   = {OFA-Diffusion Compression: Compressing Diffusion Model in One-Shot Manner},
  year    = {2025},
}
```
