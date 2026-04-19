#!/usr/bin/env python3
"""Extract standalone pruned SD v1.5 UNet subnets from an OFA-trained model.

After OFA joint training with hook-based slicing (ofa_train_sd_physical.py),
all subnets share the same full-size weight tensors and are selected at runtime
via apply_slice_hooks().  This script permanently removes the pruned channels,
producing a standalone smaller UNet for each target ratio:

  - Conv pipe:  conv1 output rows + time_emb_proj rows + norm2 channels
                + conv2 input cols are physically removed
  - Attn pipe:  to_q/to_k/to_v output rows + to_out.0 input cols are
                physically removed  (per-head-offset, all heads keep same offsets)
  - FFN pipe:   GEGLU proj output rows + ff.net.2 input cols are
                physically removed

The extracted UNet runs standard diffusers forward (no hooks needed)
with reduced internal dimensions — ready for independent fine-tuning.

Usage:
  cd /wherever/OFA/Diff-Pruning
  python sd_extract_subnet.py \
      --sd_path     pretrained/sd-v1-5 \
      --unet_path   outputs/ofa_sd_physical/train/final/unet_ema \
      --masks_path  outputs/ofa_sd_physical/masks/sd_masks.pt \
      --ratios      0.0,0.5,1.0 \
      --outdir      outputs/extracted_subnets_sd
"""

import os
import sys
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure Diff-Pruning is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from diffusers import UNet2DConditionModel
from networks_ofa import (
    build_subnet_cfg_from_masks,
    _per_head_rows, _per_head_geglu_rows,
)


# ===========================================================================
# Physical pruning helpers
# ===========================================================================

def _prune_resnet_conv_pipe(resnet, conv_keep_idx):
    """Physically prune the conv pipe of a ResnetBlock2D.

    Conv pipe:
      conv1:         [out_ch, in_ch, k, k] → keep output rows
      time_emb_proj: [out_ch, emb_ch]      → keep output rows
      norm2:         [out_ch]               → keep channels
      conv2:         [out_ch, out_ch, k, k] → keep input cols
    """
    keep = conv_keep_idx
    ck = keep.numel()
    out_ch = resnet.conv1.weight.shape[0]

    # ── conv1: output rows ───────────────────────────────────────────────────
    old = resnet.conv1
    new_conv1 = nn.Conv2d(old.in_channels, ck, old.kernel_size,
                          stride=old.stride, padding=old.padding,
                          dilation=old.dilation, bias=old.bias is not None)
    new_conv1.weight = nn.Parameter(old.weight[keep].clone())
    if old.bias is not None:
        new_conv1.bias = nn.Parameter(old.bias[keep].clone())
    resnet.conv1 = new_conv1

    # ── time_emb_proj: output rows ───────────────────────────────────────────
    if resnet.time_emb_proj is not None:
        old_tep = resnet.time_emb_proj
        new_tep = nn.Linear(old_tep.in_features, ck, bias=old_tep.bias is not None)
        new_tep.weight = nn.Parameter(old_tep.weight[keep].clone())
        if old_tep.bias is not None:
            new_tep.bias = nn.Parameter(old_tep.bias[keep].clone())
        resnet.time_emb_proj = new_tep

    # ── norm2: slice channels ────────────────────────────────────────────────
    old_norm2 = resnet.norm2
    ng = old_norm2.num_groups
    while ck % ng != 0 and ng > 1:
        ng -= 1
    new_norm2 = nn.GroupNorm(ng, ck, eps=old_norm2.eps, affine=old_norm2.affine)
    if old_norm2.affine:
        new_norm2.weight = nn.Parameter(old_norm2.weight[keep].clone())
        new_norm2.bias = nn.Parameter(old_norm2.bias[keep].clone())
    resnet.norm2 = new_norm2

    # ── conv2: input cols ────────────────────────────────────────────────────
    old = resnet.conv2
    new_conv2 = nn.Conv2d(ck, old.out_channels, old.kernel_size,
                          stride=old.stride, padding=old.padding,
                          dilation=old.dilation, bias=old.bias is not None)
    new_conv2.weight = nn.Parameter(old.weight[:, keep].clone())
    if old.bias is not None:
        new_conv2.bias = nn.Parameter(old.bias.clone())
    resnet.conv2 = new_conv2

    return ck


def _prune_attn_pipe(tb, attn_keep_idx, num_heads, head_dim):
    """Physically prune the attention pipe of a BasicTransformerBlock.

    For both self-attn (attn1) and cross-attn (attn2):
      to_q/to_k/to_v: [inner_dim, in_dim] → keep output rows per-head
      to_out.0:        [inner_dim, inner_dim] → keep input cols per-head

    Input/output dimensions (inner_dim = full width) are preserved for residual
    connections.  Only the *internal* Q/K/V dimension is pruned.
    """
    keep = attn_keep_idx
    ak = keep.numel()
    attn_rows = _per_head_rows(keep, num_heads, head_dim)  # [nh*ak]

    for attn_mod in [tb.attn1, tb.attn2]:
        if attn_mod is None:
            continue

        in_dim = attn_mod.to_q.weight.shape[1]

        # to_q, to_k, to_v: keep output rows
        for name in ['to_q', 'to_k', 'to_v']:
            old = getattr(attn_mod, name)
            new_linear = nn.Linear(old.in_features, num_heads * ak,
                                   bias=old.bias is not None)
            new_linear.weight = nn.Parameter(old.weight[attn_rows].clone())
            if old.bias is not None:
                new_linear.bias = nn.Parameter(old.bias[attn_rows].clone())
            setattr(attn_mod, name, new_linear)

        # to_out[0]: keep input cols
        old = attn_mod.to_out[0]
        new_out = nn.Linear(num_heads * ak, old.out_features,
                            bias=old.bias is not None)
        new_out.weight = nn.Parameter(old.weight[:, attn_rows].clone())
        if old.bias is not None:
            new_out.bias = nn.Parameter(old.bias.clone())
        attn_mod.to_out[0] = new_out

        # Update heads metadata
        attn_mod.inner_dim = num_heads * ak
        attn_mod.head_dim = ak

    return ak


def _prune_ffn_pipe(tb, ffn_keep_idx, ff_inner):
    """Physically prune the FFN pipe of a BasicTransformerBlock.

    FFN pipe:
      ff.net[0].proj (GEGLU): [2*ff_inner, inner_dim] → keep output rows
      ff.net[2]:              [ff_inner, inner_dim]    → keep input cols
    """
    keep = ffn_keep_idx
    fk = keep.numel()
    geglu_rows = _per_head_geglu_rows(keep, ff_inner)  # [2*fk]

    # ── GEGLU proj: output rows ──────────────────────────────────────────────
    geglu_mod = tb.ff.net[0]  # GEGLU module
    old_proj = geglu_mod.proj
    new_proj = nn.Linear(old_proj.in_features, 2 * fk,
                         bias=old_proj.bias is not None)
    new_proj.weight = nn.Parameter(old_proj.weight[geglu_rows].clone())
    if old_proj.bias is not None:
        new_proj.bias = nn.Parameter(old_proj.bias[geglu_rows].clone())
    geglu_mod.proj = new_proj

    # ── ff.net[2] (down proj): input cols ────────────────────────────────────
    old_down = tb.ff.net[2]
    new_down = nn.Linear(fk, old_down.out_features,
                         bias=old_down.bias is not None)
    new_down.weight = nn.Parameter(old_down.weight[:, keep].clone())
    if old_down.bias is not None:
        new_down.bias = nn.Parameter(old_down.bias.clone())
    tb.ff.net[2] = new_down

    return fk


# ===========================================================================
# Subnet extraction (top level)
# ===========================================================================

def extract_sd_subnet(unet, masks, ratio):
    """Extract a standalone pruned UNet for the given ratio.

    unet:   UNet2DConditionModel (OFA-trained, full width)
    masks:  dict from sd_prune_physical.py
    ratio:  0.0 = smallest, 1.0 = full

    Returns:
        pruned_unet:    UNet2DConditionModel with permanently reduced layers
        per_layer_dims: dict — {layer_name: {'conv_internal': int,
                                             'attn_head_dim': int,
                                             'ffn_internal': int}}
    """
    pruned = copy.deepcopy(unet)
    pruned.cpu()

    subnet_cfg = build_subnet_cfg_from_masks(masks, ratio=ratio)
    named_mods = dict(pruned.named_modules())

    per_layer_dims = {}

    for layer_name, cfg in subnet_cfg.items():
        resnet = named_mods.get(layer_name)
        if resnet is None:
            continue

        dims = {}

        # ── Conv pipe ────────────────────────────────────────────────────────
        conv_keep_idx = cfg.get('conv_keep_idx')
        if conv_keep_idx is not None:
            ck = _prune_resnet_conv_pipe(resnet, conv_keep_idx)
            dims['conv_internal'] = ck

        # ── Attn + FFN pipe ──────────────────────────────────────────────────
        attn_keep_idx = cfg.get('attn_keep_idx')
        ffn_keep_idx = cfg.get('ffn_keep_idx')

        if attn_keep_idx is not None or ffn_keep_idx is not None:
            # Find the corresponding transformer block
            parts = layer_name.split('.')
            if 'resnets' in parts:
                idx = parts.index('resnets')
                attn_parts = parts[:idx] + ['attentions'] + parts[idx+1:]
                tb_path = '.'.join(attn_parts) + '.transformer_blocks.0'
                tb = named_mods.get(tb_path)

                if tb is not None:
                    attn1 = tb.attn1
                    num_heads = attn1.heads
                    inner_dim = attn1.to_q.weight.shape[0]
                    head_dim = inner_dim // num_heads

                    ff_proj = tb.ff.net[0].proj
                    ff_inner = ff_proj.weight.shape[0] // 2

                    if attn_keep_idx is not None:
                        ak = _prune_attn_pipe(tb, attn_keep_idx, num_heads, head_dim)
                        dims['attn_head_dim'] = ak

                    if ffn_keep_idx is not None:
                        fk = _prune_ffn_pipe(tb, ffn_keep_idx, ff_inner)
                        dims['ffn_internal'] = fk

        if dims:
            per_layer_dims[layer_name] = dims

    return pruned, per_layer_dims


def reshape_sd_to_pruned(unet, per_layer_dims, masks=None, ratio=None):
    """Reshape a standard UNet to match pruned architecture dimensions.

    Creates new Linear/Conv2d/GroupNorm with target sizes but does NOT copy
    weights.  Call unet.load_state_dict() afterwards to populate weights.

    This is used by the fine-tuning script to recreate the pruned architecture.
    """
    named_mods = dict(unet.named_modules())

    for layer_name, dims in per_layer_dims.items():
        resnet = named_mods.get(layer_name)
        if resnet is None:
            continue

        # Conv pipe
        ck = dims.get('conv_internal')
        if ck is not None:
            old = resnet.conv1
            resnet.conv1 = nn.Conv2d(old.in_channels, ck, old.kernel_size,
                                     stride=old.stride, padding=old.padding,
                                     bias=old.bias is not None)

            if resnet.time_emb_proj is not None:
                old_tep = resnet.time_emb_proj
                resnet.time_emb_proj = nn.Linear(old_tep.in_features, ck,
                                                  bias=old_tep.bias is not None)

            old_norm2 = resnet.norm2
            ng = old_norm2.num_groups
            while ck % ng != 0 and ng > 1:
                ng -= 1
            resnet.norm2 = nn.GroupNorm(ng, ck, eps=old_norm2.eps,
                                        affine=old_norm2.affine)

            old = resnet.conv2
            resnet.conv2 = nn.Conv2d(ck, old.out_channels, old.kernel_size,
                                     stride=old.stride, padding=old.padding,
                                     bias=old.bias is not None)

        # Attn + FFN pipe
        parts = layer_name.split('.')
        if 'resnets' in parts:
            idx = parts.index('resnets')
            attn_parts = parts[:idx] + ['attentions'] + parts[idx+1:]
            tb_path = '.'.join(attn_parts) + '.transformer_blocks.0'
            tb = named_mods.get(tb_path)

            if tb is not None:
                ak = dims.get('attn_head_dim')
                if ak is not None:
                    num_heads = tb.attn1.heads
                    for attn_mod in [tb.attn1, tb.attn2]:
                        if attn_mod is None:
                            continue
                        for name in ['to_q', 'to_k', 'to_v']:
                            old = getattr(attn_mod, name)
                            setattr(attn_mod, name,
                                    nn.Linear(old.in_features, num_heads * ak,
                                              bias=old.bias is not None))
                        old_out = attn_mod.to_out[0]
                        attn_mod.to_out[0] = nn.Linear(num_heads * ak,
                                                        old_out.out_features,
                                                        bias=old_out.bias is not None)
                        attn_mod.inner_dim = num_heads * ak
                        attn_mod.head_dim = ak

                fk = dims.get('ffn_internal')
                if fk is not None:
                    geglu_mod = tb.ff.net[0]
                    old_proj = geglu_mod.proj
                    geglu_mod.proj = nn.Linear(old_proj.in_features, 2 * fk,
                                               bias=old_proj.bias is not None)
                    old_down = tb.ff.net[2]
                    tb.ff.net[2] = nn.Linear(fk, old_down.out_features,
                                             bias=old_down.bias is not None)

    return unet


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract standalone pruned SD UNet subnets from OFA model')
    parser.add_argument('--sd_path', required=True,
                        help='Path to base SD v1.5 model (for config)')
    parser.add_argument('--unet_path', required=True,
                        help='Path to OFA-trained UNet (unet_ema directory)')
    parser.add_argument('--masks_path', required=True,
                        help='Path to sd_masks.pt from sd_prune_physical.py')
    parser.add_argument('--ratios', required=True,
                        help='Comma-separated ratios to extract, e.g. "0.0,0.5,1.0"')
    parser.add_argument('--outdir', default='outputs/extracted_subnets_sd',
                        help='Output directory')
    args = parser.parse_args()

    # ── Load UNet ────────────────────────────────────────────────────────────
    print(f'Loading OFA-trained UNet from "{args.unet_path}" …')
    unet = UNet2DConditionModel.from_pretrained(args.unet_path)
    unet.eval()

    n_params_full = sum(p.numel() for p in unet.parameters())
    print(f'  Full UNet: {n_params_full/1e6:.2f}M params')

    # ── Load masks ───────────────────────────────────────────────────────────
    print(f'Loading masks from "{args.masks_path}" …')
    masks = torch.load(args.masks_path, map_location='cpu', weights_only=False)
    n_layers = sum(1 for v in masks.values() if isinstance(v, dict))
    print(f'  {n_layers} prunable layers')

    # ── Parse target ratios ──────────────────────────────────────────────────
    target_ratios = [float(r) for r in args.ratios.split(',')]

    os.makedirs(args.outdir, exist_ok=True)

    # ── Extract each subnet ──────────────────────────────────────────────────
    print(f'\n{"Ratio":>8}  {"Params":>12}  {"Reduction":>10}  {"Path"}')
    print('-' * 65)

    for ratio in target_ratios:
        pruned, per_layer_dims = extract_sd_subnet(unet, masks, ratio)
        n_params = sum(p.numel() for p in pruned.parameters())
        reduction = 1 - n_params / n_params_full

        r_str = f'{ratio:.3f}'.replace('.', 'p')
        save_dir = os.path.join(args.outdir, f'subnet_r{r_str}')
        os.makedirs(save_dir, exist_ok=True)

        # Save UNet in diffusers format
        unet_dir = os.path.join(save_dir, 'unet')
        pruned.save_pretrained(unet_dir)

        # Save metadata
        meta_path = os.path.join(save_dir, 'extraction_meta.pt')
        torch.save({
            'ratio': ratio,
            'per_layer_dims': per_layer_dims,
            'n_params': n_params,
            'n_params_full': n_params_full,
        }, meta_path)

        print(f'{ratio:>8.2f}  {n_params/1e6:>10.2f}M  {reduction:>9.1%}  {unet_dir}')

        # Verify forward pass works
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pruned.to(device).eval()
        with torch.no_grad():
            x = torch.randn(1, 4, 32, 32, device=device)
            t = torch.tensor([500], device=device)
            enc = torch.randn(1, 10, 768, device=device)
            out = pruned(x, t, encoder_hidden_states=enc).sample
            assert out.shape == (1, 4, 32, 32), \
                f'Shape mismatch: {out.shape}'
        pruned.cpu()

    print(f'\n✅ Extracted {len(target_ratios)} subnets → {args.outdir}')


if __name__ == '__main__':
    main()
