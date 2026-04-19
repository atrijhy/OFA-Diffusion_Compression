# prune.py  —  SD v1.5 OFA pruning with physical-slicing-aware importance
#
# Adapted from ddpm_prune_physical.py (EDM version) for Stable Diffusion v1.5.
#
# Key differences from EDM version:
#   - Uses Diffusers UNet2DConditionModel instead of NVlabs EDM SongUNet
#   - Q/K/V are separate Linear layers (to_q, to_k, to_v) not merged qkv
#   - Cross-attention (attn2): to_k/to_v input dim = 768 (CLIP) — NOT pruned
#   - FeedForward (GEGLU): ff.net.0.proj + ff.net.2 form another pipe
#   - Conditioning: text prompts via CLIP tokenizer/encoder
#   - Loss: latent-space noise prediction MSE
#
# Prunable pipes per "Layer" (= one ResNet + one Transformer pair):
#
#   1. Conv pipe:      conv1 (output) + conv2 (input)
#      Ranks shape: [out_channels]
#
#   2. Self-Attn pipe: attn1.to_q/to_k/to_v (output) + attn1.to_out.0 (input)
#      per-head-offset aggregation → Ranks shape: [head_dim]
#
#   3. Cross-Attn pipe: attn2.to_q (output) + attn2.to_out.0 (input)
#      to_k/to_v have input_dim=768 (CLIP frozen) but output_dim = inner_dim,
#      so their output-channel importance IS included in the per-head-offset
#      aggregation together with to_q and to_out.
#      per-head-offset aggregation → Ranks shape: [head_dim]
#      (shared with self-attn keep count since they operate on same hidden dim)
#
#   4. FFN pipe:       ff.net.0.proj (output, GEGLU) + ff.net.2 (input)
#      Ranks shape: [ff_inner_dim]  (= 4 * out_channels for GEGLU: actual = 2*4*C)
#      The keep count for FFN is derived as: ffn_keep = round(C_l * ff_ratio)
#
# Mask format:
#   {
#     'masks': {
#       P_i: {
#         'down_blocks.0.resnets.0': {'conv_keep': 256, 'attn_keep': 32, 'ffn_keep': 1024},
#         'down_blocks.3.resnets.0': {'conv_keep': 960},  # no attention
#         ...
#       }
#     },
#     'conv_internal_ranks':  {blk: LongTensor[out_ch]},
#     'attn_channel_ranks':   {blk: LongTensor[head_dim]},   # self+cross shared
#     'ffn_internal_ranks':   {blk: LongTensor[ff_inner]},
#   }
#
# Usage:
#   python sd_prune_physical.py \
#       --model_path pretrained/sd-v1-5 \
#       --coco_root data/coco2014 \
#       --save_path outputs/ofa_masks_sd \
#       --device cuda:0

import os
import sys
import math
import json
import argparse
from collections import defaultdict, OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="SD v1.5 OFA pruning (physical slicing)")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to SD v1.5 Diffusers directory (with unet/, vae/, text_encoder/)")
parser.add_argument("--coco_root", type=str, required=True,
                    help="Path to COCO 2014 root (containing train2014/ and annotations/)")
parser.add_argument("--save_path", type=str, required=True,
                    help="Output directory for masks")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for importance computation (SD is memory-heavy)")
parser.add_argument("--importance_samples", type=int, default=1024,
                    help="Number of (x0,t,text) samples for per-sample Taylor importance")
parser.add_argument("--p_min", type=float, default=0.25, help="Min channel keep ratio P_0")
parser.add_argument("--n_subnets", type=int, default=31,
                    help="Number of OFA subnetworks (P_0 to 1.0 in steps of 0.025)")
parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
parser.add_argument("--align_channels", type=str, default="none", choices=["none", "32"],
                    help="Optional channel-count alignment for kept dims. "
                         "'none' follows Formula 5 directly; '32' snaps to multiples of 32 "
                         "without violating the lower bound.")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# COCO Dataset
# ---------------------------------------------------------------------------

class COCOCaptionDataset(Dataset):
    """Simple COCO dataset that returns (image_tensor, caption_string)."""
    def __init__(self, root, split='train2014', resolution=512):
        self.img_dir = os.path.join(root, split)
        ann_file = os.path.join(root, 'annotations',
                                f'captions_{split}.json')
        with open(ann_file) as f:
            data = json.load(f)

        # Build image_id -> filename mapping
        id2file = {img['id']: img['file_name'] for img in data['images']}

        # Collect (filename, caption) pairs — one caption per image
        seen = set()
        self.samples = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id in seen:
                continue
            seen.add(img_id)
            fname = id2file[img_id]
            self.samples.append((fname, ann['caption']))

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # RGB -> [-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, caption = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        return self.transform(img), caption


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    caps = [b[1] for b in batch]
    return imgs, caps


# ---------------------------------------------------------------------------
# Setup: load SD v1.5 components
# ---------------------------------------------------------------------------

print(f"[SD Physical] Loading SD v1.5 from {args.model_path}")

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

# Load components
tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    args.model_path, subfolder="text_encoder").to(args.device).eval()
vae = AutoencoderKL.from_pretrained(
    args.model_path, subfolder="vae").to(args.device).eval()
unet = UNet2DConditionModel.from_pretrained(
    args.model_path, subfolder="unet").to(args.device)
noise_scheduler = DDPMScheduler.from_pretrained(
    args.model_path, subfolder="scheduler")

# Freeze text_encoder and vae — only UNet is analyzed
text_encoder.requires_grad_(False)
vae.requires_grad_(False)

# Dataset
print(f"[SD Physical] Loading COCO 2014 from {args.coco_root}")
dataset = COCOCaptionDataset(args.coco_root, split='train2014',
                             resolution=args.resolution)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, collate_fn=collate_fn)
print(f"[SD Physical] Dataset: {len(dataset)} images")


# ---------------------------------------------------------------------------
# Helper: encode text and images
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_text(captions):
    """Tokenize captions → CLIP text embeddings [B, 77, 768]."""
    tok = tokenizer(captions, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt")
    input_ids = tok.input_ids.to(args.device)
    return text_encoder(input_ids)[0]  # [B, 77, 768]


@torch.no_grad()
def encode_images(images):
    """Encode pixel images → latent [B, 4, H/8, W/8]."""
    latents = vae.encode(images).latent_dist.sample()
    return latents * vae.config.scaling_factor  # 0.18215


def compute_loss(unet, latents, text_emb):
    """Standard SD noise prediction loss (single-step)."""
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (bsz,), device=latents.device, dtype=torch.long)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample
    return F.mse_loss(noise_pred, noise, reduction='mean')


# ---------------------------------------------------------------------------
# Step 1: Discover prunable layers and group into "Layers"
# ---------------------------------------------------------------------------
# SD v1.5 UNet structure:
#   down_blocks.{0,1,2}.resnets.{0,1}  — ResnetBlock2D with conv1/conv2
#   down_blocks.{0,1,2}.attentions.{0,1}.transformer_blocks.0  — BasicTransformerBlock
#   down_blocks.3.resnets.{0,1}  — ResnetBlock2D only (no attention)
#   mid_block.resnets.{0,1} + mid_block.attentions.0
#   up_blocks.0.resnets.{0,1,2}  — ResnetBlock2D only
#   up_blocks.{1,2,3}.resnets.{0,1,2} + attentions.{0,1,2}
#
# We define a "Layer" as one ResnetBlock2D.  If the block also has an
# associated attention (matched by index), we group them together.
# The prunable internal width C_l = conv1.out_channels = conv2.out_channels.
#
# Pipes per Layer:
#   - conv_pipe:  conv1 output + conv2 input          → ranks[out_ch]
#   - attn_pipe:  self-attn(to_q/k/v out + to_out in) → ranks[head_dim]
#                 cross-attn(to_q out + to_out in)     shared with self-attn
#   - ffn_pipe:   ff.net.0.proj output + ff.net.2 input  → ranks[ff_inner]

print("[SD Physical] Discovering prunable layers...")

unet.requires_grad_(True)

# One forward pass to identify which params get gradients
unet.zero_grad()
it = iter(loader)
imgs_batch, caps_batch = next(it)
imgs_batch = imgs_batch.to(args.device)
text_emb = encode_text(caps_batch)
latents = encode_images(imgs_batch)
loss = compute_loss(unet, latents, text_emb)
loss.backward()

# ── Discover ResNet blocks (conv1/conv2 pairs) ──────────────────────────

# A "Layer" is identified by its resnet path: e.g. "down_blocks.0.resnets.0"
# blk_name = resnet path
# layers_dict[blk_name] = {
#   'conv1': (full_name, module),
#   'conv2': (full_name, module),
#   'attn1_to_q': ..., 'attn1_to_k': ..., 'attn1_to_v': ..., 'attn1_to_out': ...,
#   'attn2_to_q': ..., 'attn2_to_out': ...,
#   'ffn_up': ..., 'ffn_down': ...,
# }

blocks = OrderedDict()  # blk_name -> dict of sub-layers

named_mods = dict(unet.named_modules())

# --- Find ResNet conv1/conv2 ---
for name, m in unet.named_modules():
    if not hasattr(m, 'weight') or m.weight is None:
        continue
    if getattr(m.weight, 'grad', None) is None:
        continue
    typ = type(m).__name__
    if typ != 'Conv2d':
        continue

    leaf = name.split('.')[-1]
    if leaf not in ('conv1', 'conv2'):
        continue

    # Must be inside a resnets block, not downsamplers/upsamplers
    if 'resnets' not in name:
        continue

    # Get the resnet block path
    # e.g. "down_blocks.0.resnets.0.conv1" → "down_blocks.0.resnets.0"
    blk_name = name.rsplit('.', 1)[0]

    if blk_name not in blocks:
        blocks[blk_name] = {}
    blocks[blk_name][leaf] = (name, m)

# --- Find associated attention layers ---
# For each resnet, check if there's a matching attention block.
# Mapping: down_blocks.{i}.resnets.{j} ↔ down_blocks.{i}.attentions.{j}
#          up_blocks.{i}.resnets.{j}   ↔ up_blocks.{i}.attentions.{j}
#          mid_block.resnets.0          ↔ mid_block.attentions.0

for blk_name in list(blocks.keys()):
    # Derive the attention path
    # "down_blocks.0.resnets.0" → "down_blocks.0.attentions.0"
    # "mid_block.resnets.0"     → "mid_block.attentions.0"
    parts = blk_name.split('.')
    if 'resnets' in parts:
        idx = parts.index('resnets')
        attn_parts = parts[:idx] + ['attentions'] + parts[idx+1:]
        attn_base = '.'.join(attn_parts)
    else:
        continue

    # Check if the transformer block exists
    tb_path = f"{attn_base}.transformer_blocks.0"
    tb_mod = named_mods.get(tb_path)
    if tb_mod is None:
        continue  # No attention for this resnet (e.g. down_blocks.3)

    # Self-attention (attn1)
    for sub_name in ['to_q', 'to_k', 'to_v']:
        full = f"{tb_path}.attn1.{sub_name}"
        mod = named_mods.get(full)
        if mod is not None and hasattr(mod, 'weight'):
            blocks[blk_name][f'attn1_{sub_name}'] = (full, mod)

    # attn1.to_out.0
    to_out_path = f"{tb_path}.attn1.to_out.0"
    mod = named_mods.get(to_out_path)
    if mod is not None:
        blocks[blk_name]['attn1_to_out'] = (to_out_path, mod)

    # Cross-attention (attn2) — only to_q and to_out are prunable on output dim
    # to_k and to_v have input_dim=768 (CLIP), output_dim follows hidden_dim
    for sub_name in ['to_q', 'to_k', 'to_v']:
        full = f"{tb_path}.attn2.{sub_name}"
        mod = named_mods.get(full)
        if mod is not None and hasattr(mod, 'weight'):
            blocks[blk_name][f'attn2_{sub_name}'] = (full, mod)

    to_out2_path = f"{tb_path}.attn2.to_out.0"
    mod = named_mods.get(to_out2_path)
    if mod is not None:
        blocks[blk_name]['attn2_to_out'] = (to_out2_path, mod)

    # FeedForward (GEGLU: ff.net.0.proj + ff.net.2)
    ffn_up_path = f"{tb_path}.ff.net.0.proj"
    ffn_down_path = f"{tb_path}.ff.net.2"
    ffn_up = named_mods.get(ffn_up_path)
    ffn_down = named_mods.get(ffn_down_path)
    if ffn_up is not None and hasattr(ffn_up, 'weight'):
        blocks[blk_name]['ffn_up'] = (ffn_up_path, ffn_up)
    if ffn_down is not None and hasattr(ffn_down, 'weight'):
        blocks[blk_name]['ffn_down'] = (ffn_down_path, ffn_down)

    # proj_in / proj_out (Spatial Transformer entry/exit 1x1 conv)
    proj_in_path = f"{attn_base}.proj_in"
    proj_out_path = f"{attn_base}.proj_out"
    proj_in = named_mods.get(proj_in_path)
    proj_out = named_mods.get(proj_out_path)
    if proj_in is not None and hasattr(proj_in, 'weight'):
        blocks[blk_name]['proj_in'] = (proj_in_path, proj_in)
    if proj_out is not None and hasattr(proj_out, 'weight'):
        blocks[blk_name]['proj_out'] = (proj_out_path, proj_out)

# Get num_heads for attention blocks
def get_num_heads(blk_name):
    """Get number of attention heads for the attention associated with this resnet."""
    parts = blk_name.split('.')
    idx = parts.index('resnets')
    attn_parts = parts[:idx] + ['attentions'] + parts[idx+1:]
    attn_base = '.'.join(attn_parts)
    tb_path = f"{attn_base}.transformer_blocks.0.attn1"
    attn_mod = named_mods.get(tb_path)
    if attn_mod is not None:
        return getattr(attn_mod, 'heads', 8)
    return 8

print(f"[SD Physical] Found {len(blocks)} layers:")
for blk, layers in blocks.items():
    layer_keys = sorted(layers.keys())
    print(f"  {blk}: {', '.join(layer_keys)}")


# ---------------------------------------------------------------------------
# Step 2: Per-sample Taylor importance (Formula 4)
# ---------------------------------------------------------------------------
# Collect ALL prunable layers for output-channel importance.
# Additionally collect conv2 / attn1_to_out / attn2_to_out / ffn_down
# for INPUT-channel importance (back end of each pipe).

# Build flat lists
all_layer_names = []
all_layer_modules = []
for blk_name, layers in blocks.items():
    for leaf, (name, m) in layers.items():
        all_layer_names.append(name)
        all_layer_modules.append(m)

# Output-channel importance for every layer
def get_out_dim(m):
    return getattr(m, 'out_channels', getattr(m, 'out_features', None))

channel_imp = {}
for name, m in zip(all_layer_names, all_layer_modules):
    od = get_out_dim(m)
    if od is not None:
        channel_imp[name] = torch.zeros(od, device=args.device)

# Input-channel importance for pipe back-ends
# conv2: back of conv pipe
# attn1_to_out: back of self-attn pipe
# attn2_to_out: back of cross-attn pipe
# ffn_down: back of FFN pipe
# proj_out: back of spatial transformer pipe
input_imp = {}  # name -> Tensor[in_channels]

for blk_name, layers in blocks.items():
    for back_key in ['conv2', 'attn1_to_out', 'attn2_to_out', 'ffn_down', 'proj_out']:
        if back_key in layers:
            fname, mod = layers[back_key]
            in_ch = mod.weight.shape[1]
            input_imp[fname] = torch.zeros(in_ch, device=args.device)

print(f"\n[SD Physical] Computing per-sample Taylor importance ({args.importance_samples} samples)...")

it2 = iter(loader)
for sample_idx in tqdm(range(args.importance_samples), desc="[SD Physical] importance"):
    try:
        imgs, caps = next(it2)
    except StopIteration:
        it2 = iter(loader)
        imgs, caps = next(it2)

    imgs = imgs.to(args.device)[:1]  # single sample for accurate per-sample Taylor
    caps = caps[:1]

    with torch.no_grad():
        text_emb = encode_text(caps)
        latents = encode_images(imgs)

    unet.zero_grad()
    loss = compute_loss(unet, latents, text_emb)
    loss.backward()

    # Per output channel importance
    for name, m in zip(all_layer_names, all_layer_modules):
        if m.weight.grad is None:
            continue
        wg = m.weight.detach() * m.weight.grad
        if m.weight.dim() == 4:
            channel_imp[name] += wg.sum(dim=(1, 2, 3)).abs()
        elif m.weight.dim() == 2:
            channel_imp[name] += wg.sum(dim=1).abs()

    # Per input channel importance (back-end layers)
    for fname in input_imp:
        mod = named_mods[fname.rsplit('.', 1)[0]] if '.' in fname else None
        # Look up module directly
        for blk_name, layers in blocks.items():
            for back_key in ['conv2', 'attn1_to_out', 'attn2_to_out', 'ffn_down', 'proj_out']:
                if back_key in layers and layers[back_key][0] == fname:
                    mod = layers[back_key][1]
                    break
            else:
                continue
            break
        if mod is None or mod.weight.grad is None:
            continue
        wg = mod.weight.detach() * mod.weight.grad
        if mod.weight.dim() == 4:
            input_imp[fname] += wg.sum(dim=(0, 2, 3)).abs()
        elif mod.weight.dim() == 2:
            input_imp[fname] += wg.sum(dim=0).abs()

# Normalize
for name in channel_imp:
    channel_imp[name] /= args.importance_samples
for name in input_imp:
    input_imp[name] /= args.importance_samples


# ---------------------------------------------------------------------------
# Step 3: Aggregate importance per PIPE
# ---------------------------------------------------------------------------

conv_internal_imp = {}     # blk_name -> Tensor[out_ch]
conv_internal_ranks = {}   # blk_name -> LongTensor sorted desc
attn_channel_imp = {}      # blk_name -> Tensor[head_dim]  (self+cross combined)
attn_channel_ranks = {}    # blk_name -> LongTensor sorted desc
ffn_internal_imp = {}      # blk_name -> Tensor[ff_inner_dim]
ffn_internal_ranks = {}    # blk_name -> LongTensor sorted desc

for blk_name, layers in blocks.items():
    out_ch = None  # internal channel width for this Layer

    # ── Conv pipe: conv1 output + conv2 input ──────────────────────────
    if 'conv1' in layers and 'conv2' in layers:
        name1, m1 = layers['conv1']
        name2, m2 = layers['conv2']
        out_ch = m1.out_channels  # = m2.out_channels = internal width

        imp_conv1_out = channel_imp.get(name1, torch.zeros(out_ch, device=args.device))
        imp_conv2_in = input_imp.get(name2, torch.zeros(out_ch, device=args.device))
        pipe_imp = imp_conv1_out + imp_conv2_in
        conv_internal_imp[blk_name] = pipe_imp
        conv_internal_ranks[blk_name] = pipe_imp.argsort(descending=True).cpu()
    elif 'conv1' in layers:
        name1, m1 = layers['conv1']
        out_ch = m1.out_channels
        conv_internal_imp[blk_name] = channel_imp.get(name1, torch.zeros(out_ch, device=args.device))
        conv_internal_ranks[blk_name] = conv_internal_imp[blk_name].argsort(descending=True).cpu()

    # ── Attention pipe: self-attn + cross-attn (per-head-offset) ───────
    if 'attn1_to_q' in layers:
        nh = get_num_heads(blk_name)
        _, m_q = layers['attn1_to_q']
        inner_dim = m_q.out_features  # = nh * hd
        hd = inner_dim // nh

        # Self-attention: per-head-offset importance from Q, K, V output
        offset_imp = torch.zeros(hd, device=args.device)
        for sub in ['attn1_to_q', 'attn1_to_k', 'attn1_to_v']:
            if sub in layers:
                fname_sub = layers[sub][0]
                imp_vec = channel_imp.get(fname_sub, torch.zeros(inner_dim, device=args.device))
                for h in range(nh):
                    offset_imp += imp_vec[h*hd:(h+1)*hd]

        # Self-attention to_out input importance
        if 'attn1_to_out' in layers:
            fname_out = layers['attn1_to_out'][0]
            imp_out = input_imp.get(fname_out, torch.zeros(inner_dim, device=args.device))
            for h in range(nh):
                offset_imp += imp_out[h*hd:(h+1)*hd]

        # Cross-attention: to_q output + to_out input (same head_dim)
        # to_k/to_v output importance also contributes (output dim = inner_dim)
        if 'attn2_to_q' in layers:
            fname_q2 = layers['attn2_to_q'][0]
            imp_q2 = channel_imp.get(fname_q2, torch.zeros(inner_dim, device=args.device))
            for h in range(nh):
                offset_imp += imp_q2[h*hd:(h+1)*hd]

        # Cross-attn to_k/to_v: output dim = inner_dim, same per-head structure
        for sub2 in ['attn2_to_k', 'attn2_to_v']:
            if sub2 in layers:
                fname_sub2 = layers[sub2][0]
                imp_sub2 = channel_imp.get(fname_sub2, torch.zeros(inner_dim, device=args.device))
                for h in range(nh):
                    offset_imp += imp_sub2[h*hd:(h+1)*hd]

        if 'attn2_to_out' in layers:
            fname_out2 = layers['attn2_to_out'][0]
            imp_out2 = input_imp.get(fname_out2, torch.zeros(inner_dim, device=args.device))
            for h in range(nh):
                offset_imp += imp_out2[h*hd:(h+1)*hd]

        attn_channel_imp[blk_name] = offset_imp  # [hd]
        attn_channel_ranks[blk_name] = offset_imp.argsort(descending=True).cpu()

    # ── FFN pipe: ff.net.0.proj output + ff.net.2 input ────────────────
    if 'ffn_up' in layers and 'ffn_down' in layers:
        fname_up = layers['ffn_up'][0]
        fname_down = layers['ffn_down'][0]
        _, m_up = layers['ffn_up']
        _, m_down = layers['ffn_down']

        # GEGLU: ff.net.0.proj has out_features = 2 * ff_inner_dim (gate + value)
        # We need to handle the GEGLU split:
        # Output of ff.net.0.proj: [2*ff_inner, in] → first half is gate, second is value
        # Both halves contribute. ff.net.2 input dim = ff_inner (after GEGLU)
        total_out = m_up.out_features   # = 2 * ff_inner (GEGLU)
        ff_inner = total_out // 2       # actual working dim after GEGLU
        # ff.net.2 weight: [out_ch, ff_inner]

        # FFN up output importance (both gate and value halves)
        imp_up = channel_imp.get(fname_up, torch.zeros(total_out, device=args.device))
        # Combine gate + value for the same inner channel
        imp_up_combined = imp_up[:ff_inner] + imp_up[ff_inner:]  # [ff_inner]

        # FFN down input importance
        imp_down_in = input_imp.get(fname_down, torch.zeros(ff_inner, device=args.device))

        ffn_pipe_imp = imp_up_combined + imp_down_in
        ffn_internal_imp[blk_name] = ffn_pipe_imp
        ffn_internal_ranks[blk_name] = ffn_pipe_imp.argsort(descending=True).cpu()

print(f"\n[SD Physical] Importance computed:")
print(f"  Conv pipes:  {len(conv_internal_imp)} layers")
print(f"  Attn pipes:  {len(attn_channel_imp)} layers  (self+cross, per-head-offset)")
print(f"  FFN pipes:   {len(ffn_internal_imp)} layers")


# ---------------------------------------------------------------------------
# Step 3b: Per-Layer importance I_l^L (for Formula 5)
# ---------------------------------------------------------------------------

layer_imp = {}   # blk_name -> float
layer_size = {}  # blk_name -> int (|l| reference width axis for Formula 5)

for blk_name in blocks:
    imp = 0.0
    size = 0
    if blk_name in conv_internal_imp:
        imp += conv_internal_imp[blk_name].sum().item()
        size = conv_internal_imp[blk_name].shape[0]
    if blk_name in attn_channel_imp:
        imp += attn_channel_imp[blk_name].sum().item()
    if blk_name in ffn_internal_imp:
        imp += ffn_internal_imp[blk_name].sum().item()
    layer_imp[blk_name] = imp
    layer_size[blk_name] = size


# ---------------------------------------------------------------------------
# Step 4: Formula 5 — budget allocation per resolution group
# ---------------------------------------------------------------------------
# SD v1.5 resolution groups:
#   down_64x64:  down_blocks.0.resnets.{0,1}   C=320
#   down_32x32:  down_blocks.1.resnets.{0,1}   C=640
#   down_16x16:  down_blocks.2.resnets.{0,1}   C=1280
#   down_8x8:    down_blocks.3.resnets.{0,1}   C=1280  (no attn)
#   mid_8x8:     mid_block.resnets.{0,1}       C=1280
#   up_8x8:      up_blocks.0.resnets.{0,1,2}   C=1280  (no attn)
#   up_16x16:    up_blocks.1.resnets.{0,1,2}   C=1280
#   up_32x32:    up_blocks.2.resnets.{0,1,2}   C=640
#   up_64x64:    up_blocks.3.resnets.{0,1,2}   C=320

res_groups = OrderedDict()

for blk_name in blocks:
    parts = blk_name.split('.')
    # "down_blocks.0.resnets.0" → down_0
    # "mid_block.resnets.0"     → mid
    # "up_blocks.1.resnets.0"   → up_1
    if parts[0] == 'down_blocks':
        group_key = f"down_{parts[1]}"
    elif parts[0] == 'mid_block':
        group_key = "mid"
    elif parts[0] == 'up_blocks':
        group_key = f"up_{parts[1]}"
    else:
        group_key = "other"
    res_groups.setdefault(group_key, []).append(blk_name)

print("\n[SD Physical] Resolution groups (paper's Block B):")
for gk, layer_names in res_groups.items():
    sizes = [layer_size[n] for n in layer_names]
    print(f"  {gk}: K={len(layer_names)} layers, C={sizes}")

# ── Apply Formula 5 ──────────────────────────────────────────────────────

P_values = [round(args.p_min + i * 0.025, 4)
            for i in range(args.n_subnets)]
# Ensure 1.0 is included
if P_values[-1] < 1.0:
    P_values.append(1.0)

all_masks = {}
print("\nComputing SD OFA masks (Algorithm 1, Formula 5, per-resolution-group):")

for P_i in P_values:
    mask = {}

    if P_i >= 1.0:
        for blk_name in blocks:
            blk_cfg = {}
            if blk_name in conv_internal_imp:
                blk_cfg['conv_keep'] = conv_internal_imp[blk_name].shape[0]
            if blk_name in attn_channel_imp:
                blk_cfg['attn_keep'] = attn_channel_imp[blk_name].shape[0]
            if blk_name in ffn_internal_imp:
                blk_cfg['ffn_keep'] = ffn_internal_imp[blk_name].shape[0]
            mask[blk_name] = blk_cfg
        all_masks[P_i] = mask
        print(f"  P_i={P_i:.4f}  avg_keep=1.000  min=1.000")
        continue

    for group_key, layer_names in res_groups.items():
        K = len(layer_names)
        group_total_imp = sum(layer_imp[n] for n in layer_names)
        if group_total_imp < 1e-12:
            group_total_imp = 1e-12

        for blk_name in layer_names:
            l_size = layer_size[blk_name]
            if l_size == 0:
                continue
            l_imp = layer_imp[blk_name]

            # Formula 5
            norm_l = l_imp / group_total_imp
            C_l_raw = norm_l * K * P_i * l_size
            c_min = max(1, round(args.p_min * l_size))
            C_l = max(min(round(C_l_raw), l_size), c_min)

            # Optional 32-alignment (engineering choice, not required by Formula 5).
            # Keep formula lower-bound semantics by never going below c_min.
            if args.align_channels == "32" and C_l < l_size:
                aligned = (C_l // 32) * 32
                if aligned < c_min:
                    aligned = ((c_min + 31) // 32) * 32
                C_l = min(max(aligned, c_min), l_size)

            blk_cfg = {}
            # B-axis semantics: C_l lives on layer reference axis |l|=l_size.
            # Convert to each pipe axis using ratio = C_l / |l|.
            ratio = C_l / l_size if l_size > 0 else 1.0

            # Conv pipe mapping (legacy equivalent kept in comment).
            if blk_name in conv_internal_imp:
                conv_full = conv_internal_imp[blk_name].shape[0]
                conv_keep = max(1, round(ratio * conv_full))
                conv_keep = min(conv_keep, conv_full)
                blk_cfg['conv_keep'] = conv_keep
                # --- Legacy executable code (kept as commented source) ---
                # blk_cfg['conv_keep'] = C_l

            # Attn pipe mapping to per-head-offset axis.
            # Legacy round(C_l/nh) is equivalent because hd = |l|/nh.
            if blk_name in attn_channel_imp:
                hd = attn_channel_imp[blk_name].shape[0]
                attn_keep = max(1, round(ratio * hd))
                attn_keep = min(attn_keep, hd)
                blk_cfg['attn_keep'] = attn_keep
                # --- Legacy executable code (kept as commented source) ---
                # nh = l_size // hd if hd > 0 else 1
                # attn_keep = max(1, round(C_l / nh))
                # attn_keep = min(attn_keep, hd)
                # blk_cfg['attn_keep'] = attn_keep

            # FFN pipe: keep proportional to C_l
            # Original ratio: ff_inner = 4 * out_ch (for GEGLU, the proj outputs 8*C, inner=4*C)
            if blk_name in ffn_internal_imp:
                ff_full = ffn_internal_imp[blk_name].shape[0]
                ffn_keep = max(1, round(ratio * ff_full))
                # --- Legacy executable code (kept as commented source) ---
                # ratio = C_l / l_size
                # ffn_keep = max(1, round(ratio * ff_full))
                if args.align_channels == "32" and ffn_keep < ff_full:
                    ffn_keep = max(32, (ffn_keep // 32) * 32)
                    ffn_keep = min(ffn_keep, ff_full)
                blk_cfg['ffn_keep'] = ffn_keep

            mask[blk_name] = blk_cfg

    all_masks[P_i] = mask

    # Stats
    ratios = []
    for blk_name in blocks:
        if blk_name not in mask:
            continue
        cfg = mask[blk_name]
        if 'conv_keep' in cfg and blk_name in conv_internal_imp:
            ratios.append(cfg['conv_keep'] / conv_internal_imp[blk_name].shape[0])
    avg = sum(ratios) / len(ratios) if ratios else 1.0
    mn = min(ratios) if ratios else 1.0
    print(f"  P_i={P_i:.4f}  avg_keep={avg:.3f}  min={mn:.3f}")


# ---------------------------------------------------------------------------
# Step 5: Save
# ---------------------------------------------------------------------------

os.makedirs(args.save_path, exist_ok=True)
save_path = os.path.join(args.save_path, "ofa_masks_sd.pt")
torch.save({
    'masks': all_masks,
    'conv_internal_ranks': conv_internal_ranks,
    'attn_channel_ranks': attn_channel_ranks,
    'ffn_internal_ranks': ffn_internal_ranks,
}, save_path)

print(f"\nSaved {len(P_values)} subnetwork masks → {save_path}")
print(f"  conv_internal_ranks:  {len(conv_internal_ranks)} layers")
print(f"  attn_channel_ranks:   {len(attn_channel_ranks)} layers  (per-head-offset)")
print(f"  ffn_internal_ranks:   {len(ffn_internal_ranks)} layers")
print(f"\nUsage:")
print(f"  python ofa_train_sd_physical.py \\")
print(f"      --model_path {args.model_path} \\")
print(f"      --coco_root {args.coco_root} \\")
print(f"      --masks {save_path} \\")
print(f"      --outdir outputs/ofa_sd_trained")
