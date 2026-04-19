#!/usr/bin/env python3
"""Fine-tune an extracted (physically pruned) SD v1.5 UNet subnet.

After OFA joint training + subnet extraction (sd_extract_subnet.py), this
script fine-tunes the standalone pruned UNet with standard noise prediction
MSE loss — no OFA sandwich sampling, no slicing hooks, no subnet_cfg.  The
pruned UNet runs standard diffusers forward with permanently reduced internal
dimensions.

This is the final step in the OFA-Diffusion pipeline — each subnet is
independently fine-tuned for a small number of additional steps to recover
quality lost during joint training.

Prerequisites:
  1. Extracted subnet from sd_extract_subnet.py (diffusers format).

Usage:
  cd /wherever/OFA/Diff-Pruning
  python sd_finetune_subnet.py \
      --sd_path       pretrained/sd-v1-5 \
      --unet_path     outputs/extracted_subnets_sd/subnet_r0p50/unet \
      --coco_root     data/coco2014 \
      --outdir        outputs/finetuned_sd/r0.50 \
      --batch_size    4 --lr 2e-5 --total_steps 20000

  Multi-GPU:
  torchrun --standalone --nproc_per_node=4 sd_finetune_subnet.py ...
"""

import os
import sys
import math
import argparse
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure Diff-Pruning is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from diffusers import (
    AutoencoderKL, DDPMScheduler, UNet2DConditionModel,
)
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from pathlib import Path


# ===========================================================================
# Loading helpers
# ===========================================================================

def _load_unet_state_dict(unet_path):
    """Load saved UNet state dict from a diffusers directory."""
    sf_files = sorted(glob.glob(os.path.join(unet_path, '*.safetensors')))
    if sf_files:
        try:
            from safetensors.torch import load_file as safe_load_file
        except Exception as e:
            raise RuntimeError(
                f'Found safetensors in "{unet_path}" but safetensors is unavailable: {e}'
            ) from e
        return safe_load_file(sf_files[0], device='cpu')

    bin_files = sorted(glob.glob(os.path.join(unet_path, '*.bin')))
    if bin_files:
        return torch.load(bin_files[0], map_location='cpu', weights_only=False)

    raise FileNotFoundError(
        f'No model weights found under "{unet_path}" (*.safetensors or *.bin).'
    )


def _load_pruned_unet(unet_path, sd_path, is_main=True):
    """Load UNet; if direct load fails, rebuild pruned architecture from metadata."""
    try:
        return UNet2DConditionModel.from_pretrained(unet_path)
    except Exception as e:
        parent_dir = os.path.dirname(os.path.abspath(unet_path.rstrip('/')))
        meta_path = os.path.join(parent_dir, 'extraction_meta.pt')
        if not os.path.isfile(meta_path):
            raise RuntimeError(
                'Direct UNet load failed and extraction metadata was not found.\n'
                f'  unet_path: {unet_path}\n'
                f'  expected meta: {meta_path}\n'
                f'  original error: {e}'
            ) from e

        if is_main:
            print('Direct UNet load failed; rebuilding pruned architecture from extraction metadata …')
            print(f'  meta: {meta_path}')

        meta = torch.load(meta_path, map_location='cpu', weights_only=False)
        per_layer_dims = meta.get('per_layer_dims')
        if not isinstance(per_layer_dims, dict):
            raise RuntimeError(f'Invalid extraction metadata (missing per_layer_dims): {meta_path}')

        from sd_extract_subnet import reshape_sd_to_pruned
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet')
        unet = reshape_sd_to_pruned(unet, per_layer_dims)

        state_dict = _load_unet_state_dict(unet_path)
        incompatible = unet.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                'Rebuilt pruned UNet but state_dict is still incompatible.\n'
                f'  missing_keys={len(incompatible.missing_keys)}\n'
                f'  unexpected_keys={len(incompatible.unexpected_keys)}'
            )

        if is_main:
            ratio = meta.get('ratio', None)
            if ratio is not None:
                print(f'  loaded extracted ratio={ratio:.4f}')
        return unet


# ===========================================================================
# Dataset (same as ofa_train_sd_physical.py)
# ===========================================================================

class COCOCaptionDataset(Dataset):
    """COCO 2014 train split with captions, returns (latent, text_ids)."""

    def __init__(self, root, tokenizer, image_size=512, max_length=77):
        super().__init__()
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length

        # Collect (image_path, caption) pairs
        import json
        ann_file = self.root / 'annotations' / 'captions_train2014.json'
        with open(ann_file) as f:
            anns = json.load(f)

        id_to_file = {img['id']: img['file_name'] for img in anns['images']}
        self.samples = []
        for ann in anns['annotations']:
            img_id = ann['image_id']
            fname = id_to_file.get(img_id)
            if fname:
                self.samples.append((
                    str(self.root / 'train2014' / fname),
                    ann['caption'],
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        # Load + resize + center-crop
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(self.image_size,
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        img = Image.open(img_path).convert('RGB')
        pixel_values = transform(img)  # [3, H, W] in [-1, 1]

        # Tokenize caption
        tokens = self.tokenizer(
            caption, padding='max_length', max_length=self.max_length,
            truncation=True, return_tensors='pt')
        input_ids = tokens.input_ids.squeeze(0)  # [max_length]

        return pixel_values, input_ids


# ===========================================================================
# Training
# ===========================================================================

def finetune_sd_subnet(args):
    """Standard SD noise-prediction training for an extracted (pruned) UNet."""

    # ── Distributed setup ────────────────────────────────────────────────────
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = world_size > 1

    if is_distributed:
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_main = local_rank == 0

    # ── Load components ──────────────────────────────────────────────────────
    if is_main:
        print(f'Loading UNet from "{args.unet_path}" …')
    unet = _load_pruned_unet(args.unet_path, args.sd_path, is_main=is_main)
    unet.to(device).train()
    unet.enable_gradient_checkpointing()

    n_params = sum(p.numel() for p in unet.parameters())
    if is_main:
        print(f'  UNet: {n_params/1e6:.2f}M params')

    if is_main:
        print(f'Loading VAE + CLIP from "{args.sd_path}" …')
    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder='vae')
    vae.to(device).eval().requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(args.sd_path, subfolder='text_encoder')
    text_encoder.to(device).eval().requires_grad_(False)
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder='scheduler')

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = COCOCaptionDataset(args.coco_root, tokenizer,
                                 image_size=args.image_size)
    if is_main:
        print(f'  Dataset: {len(dataset)} samples')

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # ── Optimizer + LR schedule ──────────────────────────────────────────────
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            unet.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(args.total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── DDP ──────────────────────────────────────────────────────────────────
    if is_distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[local_rank])

    # ── EMA ──────────────────────────────────────────────────────────────────
    import copy
    unet_ema = copy.deepcopy(
        unet.module if is_distributed else unet
    ).eval().requires_grad_(False)

    ema_decay = args.ema_decay

    @torch.no_grad()
    def ema_update():
        src = unet.module if is_distributed else unet
        for p_ema, p_net in zip(unet_ema.parameters(), src.parameters()):
            p_ema.data.mul_(ema_decay).add_(p_net.data, alpha=1 - ema_decay)

    # ── Encode helper ────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode_images(pixel_values):
        latents = vae.encode(pixel_values).latent_dist.sample()
        return latents * vae.config.scaling_factor

    @torch.no_grad()
    def encode_text(input_ids):
        return text_encoder(input_ids)[0]

    # Pre-compute uncond (empty string) token ids for CFG dropout
    _uncond_tokens = tokenizer(
        "", padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    )
    uncond_ids = _uncond_tokens.input_ids.to(device)  # [1, 77]

    # ── Training loop ────────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    if is_main:
        print(f'\nFine-tuning for {args.total_steps} steps …')

    global_step = 0
    if args.resume_from and os.path.isdir(args.resume_from):
        state_file = os.path.join(args.resume_from, 'training_state.pt')
        if os.path.isfile(state_file):
            state = torch.load(state_file, map_location='cpu')
            global_step = state['global_step']
            optimizer.load_state_dict(state['optimizer'])
            lr_scheduler.load_state_dict(state['lr_scheduler'])
            _src = unet.module if is_distributed else unet
            ckpt_unet = _load_pruned_unet(os.path.join(args.resume_from, 'unet'), args.sd_path, is_main=is_main)
            _src.load_state_dict(ckpt_unet.state_dict())
            ckpt_ema = _load_pruned_unet(os.path.join(args.resume_from, 'unet_ema'), args.sd_path, is_main=is_main)
            unet_ema.load_state_dict(ckpt_ema.state_dict())
            del ckpt_unet, ckpt_ema
            if is_main:
                print(f'  Resumed from step {global_step} ({args.resume_from})')
        else:
            if is_main:
                print(f'  WARNING: --resume_from given but {state_file} not found; starting from scratch')

    data_iter = None

    while global_step < args.total_steps:
        if is_distributed:
            sampler.set_epoch(global_step // len(loader))

        if data_iter is None:
            data_iter = iter(loader)

        try:
            pixel_values, input_ids = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            pixel_values, input_ids = next(data_iter)

        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)

        # CFG unconditional dropout: replace some rows with empty-string token ids
        if args.uncond_prob > 0:
            import random as _random
            mask = torch.tensor(
                [_random.random() < args.uncond_prob for _ in range(input_ids.shape[0])],
                device=device
            )
            input_ids = torch.where(mask.unsqueeze(1), uncond_ids.expand_as(input_ids), input_ids)

        # Encode to latents + text embeddings
        latents = encode_images(pixel_values)
        text_emb = encode_text(input_ids)

        # Standard noise prediction training
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # No subnet_cfg — the pruned UNet IS the subnet
        model_pred = unet(noisy_latents, timesteps,
                          encoder_hidden_states=text_emb).sample
        if noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        loss = F.mse_loss(model_pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        ema_update()

        global_step += 1

        # Logging
        if is_main and global_step % args.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'  step={global_step}/{args.total_steps}  '
                  f'loss={loss.item():.4f}  lr={lr:.2e}')

        # Save checkpoint
        if is_main and global_step % args.save_interval == 0:
            save_dir = os.path.join(args.outdir, f'checkpoint-{global_step}')
            os.makedirs(save_dir, exist_ok=True)

            # Save EMA UNet
            ema_dir = os.path.join(save_dir, 'unet_ema')
            unet_ema.save_pretrained(ema_dir)

            # Save training UNet
            train_dir = os.path.join(save_dir, 'unet')
            src = unet.module if is_distributed else unet
            src.save_pretrained(train_dir)

            # Save optimizer / scheduler state for resume
            import torch as _torch
            _torch.save({
                'global_step': global_step,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, os.path.join(save_dir, 'training_state.pt'))

            print(f'  Saved checkpoint → {save_dir}')

    # ── Final save ───────────────────────────────────────────────────────────
    if is_main:
        final_dir = os.path.join(args.outdir, 'final')
        os.makedirs(final_dir, exist_ok=True)
        unet_ema.save_pretrained(os.path.join(final_dir, 'unet_ema'))
        src = unet.module if is_distributed else unet
        src.save_pretrained(os.path.join(final_dir, 'unet'))
        print(f'\n✅ Fine-tuning complete. Final model → {final_dir}')

    if is_distributed:
        torch.distributed.destroy_process_group()


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune an extracted SD UNet subnet')
    parser.add_argument('--sd_path', required=True,
                        help='Path to base SD v1.5 model (for VAE/CLIP)')
    parser.add_argument('--unet_path', required=True,
                        help='Path to extracted UNet (diffusers format)')
    parser.add_argument('--coco_root', required=True,
                        help='Path to COCO 2014 dataset root')
    parser.add_argument('--outdir', required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd'],
                        help='Optimizer type (sgd is useful for low-memory smoke tests)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--total_steps', type=int, default=20000,
                        help='Total fine-tuning steps')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='LR warmup steps')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log every N steps')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Training image resolution')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Checkpoint dir to resume from (e.g. outdir/checkpoint-10000)')
    parser.add_argument('--uncond_prob', type=float, default=0.1,
                        help='Probability of replacing caption with empty string for CFG training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    args = parser.parse_args()

    finetune_sd_subnet(args)


if __name__ == '__main__':
    main()
