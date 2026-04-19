#!/usr/bin/env bash
# run_coco_sd_ofa.sh — OFA-Diffusion full pipeline on SD v1.5 (MS-COCO text-to-image)
#
# Prerequisites:
#   pip install -r requirements.txt
#   Pretrained SD v1.5:  pretrained/sd-v1-5   (diffusers format)
#   MS-COCO 2014:        data/coco2014
#
# Usage:
#   GPUS=4 bash scripts/run_coco_sd_ofa.sh
#   GPUS=1 bash scripts/run_coco_sd_ofa.sh   # single GPU
set -euo pipefail

GPUS="${GPUS:-4}"
OUTDIR="${OUTDIR:-outputs/coco_sd_ofa}"
SD_PATH="pretrained/sd-v1-5"
COCO_ROOT="data/coco2014"
P_VALUES="0.25,0.375,0.5,0.625,0.75,0.875,1.0"

IFS=',' read -ra _G <<< "$GPUS"; N=${#_G[@]}; FIRST=${_G[0]}

cd sd

# ── Step 1: Prune (Taylor importance) ─────────────────────────────────────────
python prune.py \
    --model_path  "../$SD_PATH" \
    --coco_root   "../$COCO_ROOT" \
    --save_path   "../$OUTDIR/masks" \
    --batch_size  4

# ── Step 2: OFA joint training ────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=$GPUS train_ofa.py \
    --sd_path     "../$SD_PATH" \
    --masks_path  "../$OUTDIR/masks/ofa_masks_physical.pt" \
    --coco_root   "../$COCO_ROOT" \
    --outdir      "../$OUTDIR/ofa_training" \
    --batch_size  16 --lr 5e-5 --total_steps 50000

# ── Step 3: Extract subnets ────────────────────────────────────────────────────
python extract.py \
    --sd_path    "../$SD_PATH" \
    --unet_path  "../$OUTDIR/ofa_training/unet_ema" \
    --masks_path "../$OUTDIR/masks/ofa_masks_physical.pt" \
    --ratios     "$P_VALUES" \
    --outdir     "../$OUTDIR/extracted"

# ── Step 4: Fine-tune each subnet ─────────────────────────────────────────────
for P in 0.25 0.375 0.5 0.625 0.75 0.875 1.0; do
    PS=$(printf '%.4f' "$P" | tr '.' 'p')
    torchrun --standalone --nproc_per_node=$GPUS finetune.py \
        --sd_path    "../$SD_PATH" \
        --unet_path  "../$OUTDIR/extracted/subnet_${PS}/unet" \
        --coco_root  "../$COCO_ROOT" \
        --outdir     "../$OUTDIR/finetuned/p${PS}" \
        --batch_size 16 --lr 2e-5 --total_steps 20000
done

# ── Step 5: Generate & evaluate FID (30k samples, 50-step DDIM) ───────────────
for P in 0.25 0.375 0.5 0.625 0.75 0.875 1.0; do
    PS=$(printf '%.4f' "$P" | tr '.' 'p')
    torchrun --standalone --nproc_per_node=$GPUS generate.py \
        --sd_path      "../$SD_PATH" \
        --unet_path    "../$OUTDIR/finetuned/p${PS}/unet_ema" \
        --coco_root    "../$COCO_ROOT" \
        --outdir       "../$OUTDIR/eval/p${PS}/samples" \
        --subnet_ratio "$P" \
        --num_images   30000 --batch_size 16 \
        --guidance_scale 7.5

    python eval_fid.py \
        --outdir "../$OUTDIR/eval/p${PS}"
done

echo "Done. FID results in $OUTDIR/eval/"
