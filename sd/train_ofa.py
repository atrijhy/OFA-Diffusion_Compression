# train_ofa.py  —  OFA training for SD v1.5 with hook-based slicing
#
# Usage (single GPU):
#   python ofa_train_sd_physical.py \
#       --sd_path pretrained/sd-v1-5 \
#       --masks_path outputs/sd_masks_physical/sd_masks.pt \
#       --coco_root data/coco2014 \
#       --outdir outputs/ofa_sd_physical \
#       --batch_size 4 --lr 5e-5 --total_steps 50000 --weight_m 3.0
#
# Usage (multi-GPU with torchrun):
#   torchrun --standalone --nproc_per_node=4 ofa_train_sd_physical.py \
#       --sd_path pretrained/sd-v1-5 \
#       --masks_path outputs/sd_masks_physical/sd_masks.pt \
#       --coco_root data/coco2014 \
#       --outdir outputs/ofa_sd_physical \
#       --batch_size 16 --lr 5e-5 --total_steps 50000 --weight_m 3.0

import os
import sys
import copy
import json
import time
import math
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

from networks_ofa import (
    sliced_forward, build_subnet_cfg_from_masks,
    sample_weighted_subnet_cfg, apply_slice_hooks, remove_slice_hooks,
)


# ---------------------------------------------------------------------------
# COCO Caption Dataset
# ---------------------------------------------------------------------------

class COCOCaptionDataset(Dataset):
    """MS-COCO 2014 image-caption dataset for SD training."""

    def __init__(self, root: str, split: str = 'train', resolution: int = 512):
        super().__init__()
        self.resolution = resolution
        self.root = Path(root)

        if split == 'train':
            self.img_dir = self.root / 'train2014'
            ann_file = self.root / 'annotations' / 'captions_train2014.json'
        else:
            self.img_dir = self.root / 'val2014'
            ann_file = self.root / 'annotations' / 'captions_val2014.json'

        assert ann_file.exists(), f"Annotation file not found: {ann_file}"
        assert self.img_dir.exists(), f"Image directory not found: {self.img_dir}"

        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Build image_id -> filename mapping
        id2file = {img['id']: img['file_name'] for img in data['images']}

        # Collect (image_path, caption) pairs
        self.samples = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id in id2file:
                img_path = self.img_dir / id2file[img_id]
                self.samples.append((str(img_path), ann['caption']))

        print(f"COCOCaptionDataset: {len(self.samples)} samples from {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback to random index on corrupted images
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Center crop and resize to resolution
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)

        # To tensor [-1, 1]
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0

        return img, caption


def collate_fn(batch):
    """Custom collate to handle string captions."""
    images = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    return images, captions


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def encode_text(tokenizer, text_encoder, captions, device, max_length=77):
    """Encode captions with CLIP tokenizer + text encoder."""
    tokens = tokenizer(
        captions,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0]
    return encoder_hidden_states


def encode_images(vae, images):
    """Encode images to latent space with VAE."""
    with torch.no_grad():
        latent_dist = vae.encode(images).latent_dist
        latents = latent_dist.sample() * vae.config.scaling_factor
    return latents


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # --- Distributed setup ---
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    device = torch.device(f'cuda:{local_rank}')
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    random.seed(args.seed + local_rank)

    # --- Load models ---
    print0(f"Loading SD v1.5 from {args.sd_path}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_path, subfolder="text_encoder").to(device)
    text_encoder.eval().requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae").to(device)
    vae.eval().requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder="unet").to(device)
    unet.train().requires_grad_(True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")

    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print0(f"  UNet: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    # --- Load OFA masks ---
    print0(f"Loading OFA masks from {args.masks_path}...")
    masks = torch.load(args.masks_path, map_location='cpu', weights_only=False)
    if isinstance(masks, dict) and 'masks' in masks:
        print0(f"  Loaded compact mask package with {len(masks['masks'])} P-values")
    else:
        print0(f"  Loaded flat mask dict with {len(masks)} layers")

    # --- EMA ---
    ema_unet = copy.deepcopy(unet).eval().requires_grad_(False)

    # --- DDP ---
    if is_distributed:
        unet_ddp = DDP(unet, device_ids=[local_rank], find_unused_parameters=True)
    else:
        unet_ddp = unet

    # --- Dataset ---
    print0(f"Loading COCO dataset from {args.coco_root}...")
    dataset = COCOCaptionDataset(args.coco_root, split='train', resolution=args.resolution)

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // world_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # --- Optimizer ---
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            unet.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-8,
        )

    # --- LR scheduler ---
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        return 1.0
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Output directory ---
    if is_main_process():
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    # --- Training loop ---
    # Paper Section 5: each step samples ONE P_i from a linearly-decreasing
    # distribution (smallest subnet sampled most often, weight_m times more
    # than the full subnet).  This avoids the sandwich strategy's over/under-fit.
    print0(f"\nStarting OFA training for {args.total_steps} steps...")
    print0(f"  Batch size: {args.batch_size} (per GPU: {args.batch_size // world_size})")
    print0(f"  Learning rate: {args.lr}")
    print0(f"  Sampling: linear-descending weights, m={args.weight_m} (paper Sec.5)")
    print0()

    global_step = 0
    if args.resume_from and os.path.isdir(args.resume_from):
        state_file = os.path.join(args.resume_from, 'training_state.pt')
        if os.path.isfile(state_file):
            state = torch.load(state_file, map_location='cpu')
            global_step = state['global_step']
            optimizer.load_state_dict(state['optimizer'])
            lr_scheduler.load_state_dict(state['lr_scheduler'])
            ckpt_unet = UNet2DConditionModel.from_pretrained(os.path.join(args.resume_from, 'unet'))
            unet.load_state_dict(ckpt_unet.state_dict())
            ckpt_ema = UNet2DConditionModel.from_pretrained(os.path.join(args.resume_from, 'unet_ema'))
            ema_unet.load_state_dict(ckpt_ema.state_dict())
            del ckpt_unet, ckpt_ema
            print0(f"  Resumed from step {global_step} ({args.resume_from})")
        else:
            print0(f"  WARNING: --resume_from given but {state_file} not found; starting from scratch")

    start_time = time.time()
    data_iter = iter(dataloader)
    log_interval = args.log_interval
    save_interval = args.save_interval
    running_loss = 0.0

    while global_step < args.total_steps:
        # Reset dataloader if exhausted
        try:
            images, captions = next(data_iter)
        except StopIteration:
            if is_distributed:
                sampler.set_epoch(global_step)
            data_iter = iter(dataloader)
            images, captions = next(data_iter)

        images = images.to(device)

        # CFG unconditional dropout: randomly replace captions with empty string
        if args.uncond_prob > 0:
            captions = [
                "" if random.random() < args.uncond_prob else c
                for c in captions
            ]

        # Encode text and images
        encoder_hidden_states = encode_text(tokenizer, text_encoder, captions, device)
        latents = encode_images(vae, images)

        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # --- Paper's training strategy: sample ONE P_i per step ---
        subnet_cfg = sample_weighted_subnet_cfg(masks, m=args.weight_m)

        raw_unet = unet_ddp.module if is_distributed else unet_ddp
        apply_slice_hooks(raw_unet, subnet_cfg)

        optimizer.zero_grad()
        try:
            # Forward pass
            model_pred = unet_ddp(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # Compute loss (noise prediction)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()

        finally:
            remove_slice_hooks(raw_unet)

        # Gradient clipping
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()

        # EMA update
        ema_decay = min(args.ema_decay, (1 + global_step) / (10 + global_step))
        with torch.no_grad():
            for p_ema, p_net in zip(ema_unet.parameters(), unet.parameters()):
                p_ema.mul_(ema_decay).add_(p_net.data, alpha=1 - ema_decay)

        running_loss += loss.item()
        global_step += 1

        # --- Logging ---
        if global_step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed
            eta = (args.total_steps - global_step) / max(steps_per_sec, 1e-8)
            lr_now = optimizer.param_groups[0]['lr']
            gpu_mem = torch.cuda.max_memory_allocated(device) / 2**30
            print0(
                f"step {global_step:6d}/{args.total_steps}  "
                f"loss {avg_loss:.4f}  "
                f"lr {lr_now:.2e}  "
                f"steps/s {steps_per_sec:.2f}  "
                f"eta {time.strftime('%Hh%Mm', time.gmtime(eta))}  "
                f"gpumem {gpu_mem:.1f}GB"
            )
            running_loss = 0.0
            torch.cuda.reset_peak_memory_stats(device)

        # --- Save checkpoint ---
        if global_step % save_interval == 0 and is_main_process():
            ckpt_path = os.path.join(args.outdir, f'checkpoint-{global_step:06d}')
            os.makedirs(ckpt_path, exist_ok=True)

            # Save UNet weights
            unet.save_pretrained(os.path.join(ckpt_path, 'unet'))
            ema_unet.save_pretrained(os.path.join(ckpt_path, 'unet_ema'))

            # Save optimizer state
            torch.save({
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'global_step': global_step,
            }, os.path.join(ckpt_path, 'training_state.pt'))

            print0(f"  Saved checkpoint to {ckpt_path}")

    # --- Final save ---
    if is_main_process():
        final_path = os.path.join(args.outdir, 'final')
        os.makedirs(final_path, exist_ok=True)
        unet.save_pretrained(os.path.join(final_path, 'unet'))
        ema_unet.save_pretrained(os.path.join(final_path, 'unet_ema'))
        print0(f"Saved final model to {final_path}")

    print0(f"\nTraining complete! Total time: {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - start_time))}")

    if is_distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='OFA Training for SD v1.5')

    # Model
    parser.add_argument('--sd_path', type=str, default='pretrained/sd-v1-5',
                        help='Path to SD v1.5 model directory')
    parser.add_argument('--masks_path', type=str, required=True,
                        help='Path to OFA masks file (from sd_prune_physical.py)')

    # Data
    parser.add_argument('--coco_root', type=str, default='data/coco2014',
                        help='Path to COCO 2014 dataset root')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Training image resolution')

    # Training
    parser.add_argument('--outdir', type=str, default='outputs/ofa_sd_physical',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Total batch size across all GPUs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd'],
                        help='Optimizer type (sgd is useful for low-memory smoke tests)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='AdamW weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 to disable)')
    parser.add_argument('--total_steps', type=int, default=50000,
                        help='Total training steps')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='LR warmup steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # OFA
    parser.add_argument('--weight_m', type=float, default=3.0,
                        help='Linear-weight ratio m: w_{P_1} = m * w_{P_N} (paper Sec.5, default m=3)')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate')

    # Logging
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log every N steps')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Checkpoint dir to resume from (e.g. outdir/checkpoint-050000)')
    parser.add_argument('--uncond_prob', type=float, default=0.1,
                        help='Probability of replacing caption with empty string for CFG training')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
