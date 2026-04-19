#!/usr/bin/env bash
# run_cifar10_ofa.sh — OFA-Diffusion full pipeline on CIFAR-10 32x32 (uncond, VP)
#
# Prerequisites:
#   pip install -r requirements.txt
#   EDM repo (NVlabs/edm) cloned at:   third_party/edm
#   Pretrained EDM model:               pretrained/edm-cifar10-32x32-uncond-vp.pkl
#   CIFAR-10 dataset zip:               data/cifar10-32x32.zip
#
# Usage:
#   GPUS=4 bash scripts/run_cifar10_ofa.sh
#   GPUS=1 bash scripts/run_cifar10_ofa.sh   # single GPU
set -euo pipefail

GPUS="${GPUS:-4}"
OUTDIR="${OUTDIR:-outputs/cifar10_ofa}"
PRETRAINED="pretrained/edm-cifar10-32x32-uncond-vp.pkl"
DATASET="data/cifar10-32x32.zip"
EDM_REPO="third_party/edm"
P_VALUES="0.25,0.375,0.5,0.625,0.75,0.875,1.0"

export PYTHONPATH="$EDM_REPO:$PYTHONPATH"

cd edm

# ── Step 1: Prune (Taylor importance) ─────────────────────────────────────────
python prune.py \
    --model_path  "../$PRETRAINED" \
    --edm_repo    "../$EDM_REPO" \
    --dataset     "../$DATASET" \
    --save_path   "../$OUTDIR/masks" \
    --precond     vp \
    --batch_size  128

# ── Step 2: OFA joint training (102.4 Mimg ≈ 200k steps at batch 512) ─────────
torchrun --standalone --nproc_per_node=$GPUS train_ofa.py \
    --outdir     "../$OUTDIR/ofa_training" \
    --data       "../$DATASET" \
    --masks      "../$OUTDIR/masks/ofa_masks_physical.pt" \
    --transfer   "../$PRETRAINED" \
    --precond    vp --arch ddpmpp \
    --batch      512 --lr 1e-3 --duration 102.4

OFA_PKL=$(ls -v "../$OUTDIR/ofa_training/"*/network-snapshot-*.pkl | tail -1)

# ── Step 3: Extract subnets ────────────────────────────────────────────────────
python extract.py \
    --network   "$OFA_PKL" \
    --masks     "../$OUTDIR/masks/ofa_masks_physical.pt" \
    --p_values  "$P_VALUES" \
    --outdir    "../$OUTDIR/extracted"

# ── Step 4: Fine-tune each subnet (10.24 Mimg) ────────────────────────────────
for P in 0.25 0.375 0.5 0.625 0.75 0.875 1.0; do
    PS=$(printf '%.4f' "$P" | tr '.' 'p')
    torchrun --standalone --nproc_per_node=$GPUS finetune.py \
        --network  "../$OUTDIR/extracted/subnet_${PS}/network.pkl" \
        --outdir   "../$OUTDIR/finetuned/p${PS}" \
        --data     "../$DATASET" \
        --precond  vp --arch ddpmpp \
        --batch    512 --lr 2e-4 --duration 10.24
done

# ── Step 5: Generate & evaluate FID (50k samples, 18-step Heun) ───────────────
for P in 0.25 0.375 0.5 0.625 0.75 0.875 1.0; do
    PS=$(printf '%.4f' "$P" | tr '.' 'p')
    FT_PKL=$(ls -v "../$OUTDIR/finetuned/p${PS}/"*/network-snapshot-*.pkl | tail -1)
    torchrun --standalone --nproc_per_node=$GPUS generate.py \
        --network  "$FT_PKL" \
        --masks    "../$OUTDIR/masks/ofa_masks_physical.pt" \
        --p_i      "$P" \
        --outdir   "../$OUTDIR/eval/p${PS}/samples" \
        --seeds    0-49999 --subdirs --batch 256 --steps 18
done

echo "Done. Samples in $OUTDIR/eval/"
