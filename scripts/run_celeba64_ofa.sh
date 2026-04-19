#!/usr/bin/env bash
# run_celeba64_ofa.sh — OFA-Diffusion full pipeline on CelebA 64x64 (uncond)
#
# Prerequisites:
#   pip install -r requirements.txt
#   Pretrained UViT:  pretrained/celeba64_uvit_small.pth
#   CelebA dataset:   data/celeba  (LMDB or raw images, see uvit/configs/celeba64_uvit_small.py)
#
# Usage:
#   GPUS=0,1,2,3 bash scripts/run_celeba64_ofa.sh
#   GPUS=0       bash scripts/run_celeba64_ofa.sh   # single GPU
set -euo pipefail

GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-29400}"
OUTDIR="${OUTDIR:-outputs/celeba64_ofa}"
CONFIG="uvit/configs/celeba64_uvit_small.py"
PRETRAINED="pretrained/celeba64_uvit_small.pth"
DATASET="data/celeba"
P_VALUES="0.25,0.375,0.5,0.625,0.75,0.875,1.0"

IFS=',' read -ra _G <<< "$GPUS"; N=${#_G[@]}; FIRST=${_G[0]}

launch() { CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --multi_gpu --num_processes $N --main_process_port $PORT "$@"; }
single()  { CUDA_VISIBLE_DEVICES=$FIRST python "$@"; }
p_str()   { printf '%.4f' "$1" | tr '.' 'p'; }

cd uvit

# ── Step 1: Prune (Taylor importance) ─────────────────────────────────────────
single prune.py \
    --config  "../$CONFIG" \
    --ckpt    "../$PRETRAINED" \
    --dataset_path "../$DATASET" \
    --outdir  "../$OUTDIR/masks"

# ── Step 2: OFA joint training (200k steps) ───────────────────────────────────
launch train_ofa.py \
    --config="../$CONFIG" \
    --masks="../$OUTDIR/masks/uvit_masks.pt" \
    --transfer="../$PRETRAINED" \
    --workdir="../$OUTDIR/ofa_training" \
    --config.train.n_steps=200000 \
    --config.train.batch_size=512 \
    --config.optimizer.lr=1e-4 \
    --config.dataset.path="../$DATASET"

OFA_CKPT=$(ls -v "../$OUTDIR/ofa_training/ckpts/"*.ckpt | tail -1)

# ── Step 3: Extract subnets ────────────────────────────────────────────────────
single extract.py \
    --config   "../$CONFIG" \
    --ckpt     "$OFA_CKPT" \
    --masks    "../$OUTDIR/masks/uvit_masks.pt" \
    --p_values "$P_VALUES" \
    --outdir   "../$OUTDIR/extracted"

# ── Step 4: Fine-tune each subnet (40k steps) ─────────────────────────────────
for P in 0.25 0.375 0.5 0.625 0.75 0.875 1.0; do
    PS=$(p_str "$P")
    launch finetune.py \
        --config="../$CONFIG" \
        --extracted="../$OUTDIR/extracted/subnet_${PS}/model.pth" \
        --n_steps=40000 \
        --save_steps="10000,20000,30000,40000" \
        --workdir="../$OUTDIR/finetuned/p${PS}" \
        --config.train.batch_size=256 \
        --config.optimizer.lr=1e-4 \
        --dataset_path="../$DATASET"
done

# ── Step 5: Evaluate FID (50k samples, 50-step DPM-Solver) ───────────────────
for P in 0.25 0.375 0.5 0.625 0.75 0.875 1.0; do
    PS=$(p_str "$P")
    launch eval_fid.py \
        --config="../$CONFIG" \
        --extracted="../$OUTDIR/extracted/subnet_${PS}/model.pth" \
        --ckpt="../$OUTDIR/finetuned/p${PS}/ckpts/40000.ckpt" \
        --n_samples=50000 \
        --sample_steps=50 \
        --algorithm=dpm_solver \
        --outdir="../$OUTDIR/eval/p${PS}" \
        --dataset_path="../$DATASET"
done

echo "Done. FID results in $OUTDIR/eval/"
