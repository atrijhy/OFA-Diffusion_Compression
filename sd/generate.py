# generate.py  —  Generate images from OFA-pruned SD v1.5 for FID evaluation
#
# Supports single-GPU and multi-GPU (torchrun) generation.
# Each rank generates its own slice of images; file names use global indices.
#
# Single-GPU usage:
#   python ofa_generate_sd.py \
#       --sd_path pretrained/sd-v1-5 \
#       --unet_path outputs/ofa_sd_physical/final/unet_ema \
#       --masks_path outputs/sd_masks_physical/sd_masks.pt \
#       --coco_root data/coco2014 \
#       --outdir outputs/ofa_sd_generated \
#       --num_images 30000 --batch_size 8
#
# Multi-GPU usage (4 GPUs):
#   torchrun --standalone --nproc_per_node=4 ofa_generate_sd.py \
#       --sd_path pretrained/sd-v1-5 \
#       --unet_path outputs/ofa_sd_physical/final/unet_ema \
#       --masks_path outputs/sd_masks_physical/sd_masks.pt \
#       --coco_root data/coco2014 \
#       --outdir outputs/ofa_sd_generated \
#       --num_images 30000 --batch_size 8

import os
import json
import argparse
import random
import glob
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

from networks_ofa_sd import (
    apply_slice_hooks, remove_slice_hooks,
    build_subnet_cfg_from_masks, get_smallest_subnet_cfg,
)


# ── Distributed helpers ───────────────────────────────────────────────────────

def _init_dist():
    """Init torch.distributed if launched via torchrun, else single-process."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


def _print0(msg, rank):
    if rank == 0:
        print(msg)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images from OFA-pruned SD v1.5')

    parser.add_argument('--sd_path', type=str, default='pretrained/sd-v1-5')
    parser.add_argument('--unet_path', type=str, required=True,
                        help='Path to trained UNet checkpoint (directory with diffusion_pytorch_model.safetensors)')
    parser.add_argument('--meta_path', type=str, default='',
                        help='Optional extraction_meta.pt path for physically-pruned extracted subnet loading')
    parser.add_argument('--masks_path', type=str, default='',
                        help='Path to OFA masks file. Leave empty to disable slicing.')
    parser.add_argument('--coco_root', type=str, default='data/coco2014')
    parser.add_argument('--outdir', type=str, default='outputs/ofa_sd_generated')
    parser.add_argument('--num_images', type=int, default=30000,
                        help='Number of images to generate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of DDIM denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--subnet_ratio', type=float, default=0.0,
                        help='Subnet ratio (0.0 = smallest, 1.0 = full)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_unet_state_dict(unet_path):
    sf_files = sorted(glob.glob(os.path.join(unet_path, '*.safetensors')))
    if sf_files:
        from safetensors.torch import load_file as safe_load_file
        return safe_load_file(sf_files[0], device='cpu')
    bin_files = sorted(glob.glob(os.path.join(unet_path, '*.bin')))
    if bin_files:
        return torch.load(bin_files[0], map_location='cpu', weights_only=False)
    raise FileNotFoundError(f'No UNet weights found under "{unet_path}"')


def _load_unet_with_fallback(unet_path, sd_path, meta_path='', device='cpu'):
    """Load UNet directly; if shape mismatch, rebuild pruned architecture and load weights."""
    try:
        return UNet2DConditionModel.from_pretrained(unet_path).to(device)
    except Exception as e:
        resolved_meta = meta_path.strip()
        if not resolved_meta:
            parent_dir = os.path.dirname(os.path.abspath(unet_path.rstrip('/')))
            candidate = os.path.join(parent_dir, 'extraction_meta.pt')
            if os.path.isfile(candidate):
                resolved_meta = candidate
        if not resolved_meta or not os.path.isfile(resolved_meta):
            raise RuntimeError(
                'UNet direct load failed and no valid extraction metadata was found.\n'
                f'  unet_path={unet_path}\n'
                f'  meta_path={meta_path or "<empty>"}\n'
                f'  original_error={e}'
            ) from e

        print('Direct UNet load failed; rebuilding pruned architecture from extraction metadata...')
        print(f'  meta: {resolved_meta}')
        meta = torch.load(resolved_meta, map_location='cpu', weights_only=False)
        per_layer_dims = meta.get('per_layer_dims')
        if not isinstance(per_layer_dims, dict):
            raise RuntimeError(f'Invalid extraction meta: missing per_layer_dims in {resolved_meta}')

        from sd_extract_subnet import reshape_sd_to_pruned
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet')
        unet = reshape_sd_to_pruned(unet, per_layer_dims)
        state_dict = _load_unet_state_dict(unet_path)
        incompatible = unet.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                'Rebuilt pruned UNet but state_dict mismatch remains.\n'
                f'  missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}'
            )
        return unet.to(device)


# ── Caption loading ───────────────────────────────────────────────────────────

def load_coco_val_captions(coco_root, num_images):
    """Load val2014 captions for generation."""
    ann_file = Path(coco_root) / 'annotations' / 'captions_val2014.json'
    assert ann_file.exists(), f"Annotation file not found: {ann_file}"

    with open(ann_file, 'r') as f:
        data = json.load(f)

    # Group captions by image (use first caption per image)
    img_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_captions:
            img_captions[img_id] = ann['caption']

    captions = list(img_captions.values())

    # If we need more than available, cycle
    if num_images > len(captions):
        repeats = (num_images // len(captions)) + 1
        captions = (captions * repeats)[:num_images]
    else:
        captions = captions[:num_images]

    return captions


# ── Main generation ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate(args):
    rank, world_size = _init_dist()

    # Device: if torchrun, use assigned GPU; else use args.device
    if world_size > 1:
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device(args.device)

    # Per-rank seed for deterministic but non-identical noise across GPUs
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    # --- Load models (each rank loads its own copy) ---
    _print0(f"Loading models from {args.sd_path}...", rank)
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_path, subfolder="text_encoder").to(device)
    text_encoder.eval()

    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae").to(device)
    vae.eval()

    _print0(f"Loading trained UNet from {args.unet_path}...", rank)
    unet = _load_unet_with_fallback(
        unet_path=args.unet_path,
        sd_path=args.sd_path,
        meta_path=args.meta_path,
        device=device,
    )
    unet.eval()

    # DDIM scheduler
    scheduler = DDIMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inference_steps)

    # --- OFA slicing hooks ---
    subnet_cfg = None
    use_slicing = bool(args.masks_path)
    if use_slicing:
        _print0(f"Loading OFA masks from {args.masks_path}...", rank)
        masks = torch.load(args.masks_path, map_location='cpu', weights_only=False)
        subnet_cfg = build_subnet_cfg_from_masks(masks, ratio=args.subnet_ratio)
        for layer_name, cfg in subnet_cfg.items():
            for k, v in cfg.items():
                if isinstance(v, torch.Tensor):
                    cfg[k] = v.to(device)
        if rank == 0:
            n_conv = sum(1 for c in subnet_cfg.values() if 'conv_keep_idx' in c)
            n_attn = sum(1 for c in subnet_cfg.values() if 'attn_keep_idx' in c)
            n_ffn  = sum(1 for c in subnet_cfg.values() if 'ffn_keep_idx' in c)
            print(f"  Subnet ratio={args.subnet_ratio}: {n_conv} conv, {n_attn} attn, {n_ffn} ffn pipes sliced")
    else:
        _print0("No masks provided: using model as-is (no OFA slicing hooks).", rank)

    # --- Load and shard captions across ranks ---
    all_captions = load_coco_val_captions(args.coco_root, args.num_images)
    _print0(f"Loaded {len(all_captions)} captions from val2014", rank)

    # Each rank gets a contiguous slice: [rank_start, rank_end)
    total = len(all_captions)
    per_rank = (total + world_size - 1) // world_size
    rank_start = rank * per_rank
    rank_end   = min(rank_start + per_rank, total)
    local_captions = all_captions[rank_start:rank_end]

    _print0(f"[Rank {rank}] generating indices [{rank_start}, {rank_end}) = {len(local_captions)} images", rank)

    # --- Output directory (shared across ranks) ---
    os.makedirs(args.outdir, exist_ok=True)

    # --- Generate ---
    latent_shape = (4, args.resolution // 8, args.resolution // 8)

    if use_slicing:
        apply_slice_hooks(unet, subnet_cfg)

    try:
        local_generated = 0
        pbar = tqdm(
            range(0, len(local_captions), args.batch_size),
            desc=f"Rank {rank}",
            disable=(rank != 0),
        )
        for batch_start in pbar:
            batch_captions = local_captions[batch_start:batch_start + args.batch_size]
            bsz = len(batch_captions)

            # Encode text
            tokens = tokenizer(
                batch_captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            encoder_hidden_states = text_encoder(tokens)[0]

            # Unconditional embeddings for CFG
            uncond_tokens = tokenizer(
                [""] * bsz,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            uncond_hidden_states = text_encoder(uncond_tokens)[0]

            text_embeddings = torch.cat([uncond_hidden_states, encoder_hidden_states])

            # Initial noise
            latents = torch.randn(bsz, *latent_shape, device=device)
            latents = latents * scheduler.init_noise_sigma

            # Denoising loop
            for t in scheduler.timesteps:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Decode
            latents_scaled = latents / vae.config.scaling_factor
            images = vae.decode(latents_scaled).sample
            images = (images / 2 + 0.5).clamp(0, 1)

            # Save with global index to avoid filename collisions
            for i in range(bsz):
                global_idx = rank_start + local_generated
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(args.outdir, f'{global_idx:06d}.png'))
                local_generated += 1

    finally:
        if use_slicing:
            remove_slice_hooks(unet)

    _print0(f"\nRank {rank}: generated {local_generated} images", rank)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    _print0(f"All ranks done. Images saved to {args.outdir}", rank)


if __name__ == '__main__':
    args = parse_args()
    generate(args)
