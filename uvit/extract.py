#!/usr/bin/env python3
"""Extract standalone pruned U-ViT subnets from an OFA-trained model.

After OFA joint training, all subnets share the same full-size weight tensors
and are selected at runtime via physical slicing (F.linear on weight sub-rows).
This script permanently removes the pruned channels, producing a standalone
smaller model for each target P_i:

  - FFN: fc1 rows + fc2 cols are physically removed → smaller hidden_dim
  - Attention: qkv rows + proj cols are physically removed → smaller head_dim
    (per-head-offset, all heads keep same offsets)

The extracted model runs the "full" forward path (no subnet_cfg needed)
with reduced internal dimensions — ready for independent fine-tuning.

Usage:
  python uvit_extract_subnet.py \
      --config    configs/cifar10_uvit_small.py \
      --ckpt      workdir/ckpts/200000.ckpt \
      --masks     outputs/uvit_masks/uvit_masks.pt \
      --p_values  0.25,0.5,0.75 \
      --outdir    outputs/extracted_subnets
"""

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

# Ensure U-ViT root is importable
_UVIT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _UVIT_ROOT not in sys.path:
    sys.path.insert(0, _UVIT_ROOT)

import utils
from libs.uvit import _per_head_qkv_rows, _per_head_proj_cols


# ===========================================================================
# Subnet extraction  (physical pruning — permanent weight removal)
# ===========================================================================

def _get_block(model, blk_name):
    """Resolve 'in_blocks.0' / 'mid_block' / 'out_blocks.2' → nn.Module."""
    parts = blk_name.split('.')
    if parts[0] == 'in_blocks':
        return model.in_blocks[int(parts[1])]
    elif parts[0] == 'mid_block':
        return model.mid_block
    elif parts[0] == 'out_blocks':
        return model.out_blocks[int(parts[1])]
    else:
        raise ValueError(f'Unknown block name: {blk_name}')


def _prune_ffn(mlp, keep_idx):
    """Replace fc1/fc2 with smaller Linear layers keeping only `keep_idx`."""
    old_fc1, old_fc2 = mlp.fc1, mlp.fc2
    in_f  = old_fc1.in_features
    out_f = old_fc2.out_features
    k     = keep_idx.numel()

    new_fc1 = nn.Linear(in_f, k, bias=old_fc1.bias is not None)
    new_fc1.weight.data.copy_(old_fc1.weight.data[keep_idx])
    if old_fc1.bias is not None:
        new_fc1.bias.data.copy_(old_fc1.bias.data[keep_idx])

    new_fc2 = nn.Linear(k, out_f, bias=old_fc2.bias is not None)
    new_fc2.weight.data.copy_(old_fc2.weight.data[:, keep_idx])
    if old_fc2.bias is not None:
        new_fc2.bias.data.copy_(old_fc2.bias.data)

    mlp.fc1 = new_fc1
    mlp.fc2 = new_fc2


def _prune_attn(attn, keep_idx):
    """Replace qkv/proj with smaller Linear layers keeping `keep_idx` offsets."""
    nh = attn.num_heads
    hd = attn.head_dim
    kd = keep_idx.numel()

    qkv_rows  = _per_head_qkv_rows(keep_idx, nh, hd)    # [3*nh*kd]
    proj_cols = _per_head_proj_cols(keep_idx, nh, hd)     # [nh*kd]

    embed_dim = attn.proj.out_features
    old_qkv   = attn.qkv
    old_proj  = attn.proj

    # QKV: [3*nh*hd, embed_dim] → [3*nh*kd, embed_dim]
    new_qkv = nn.Linear(embed_dim, 3 * nh * kd, bias=old_qkv.bias is not None)
    new_qkv.weight.data.copy_(old_qkv.weight.data[qkv_rows])
    if old_qkv.bias is not None:
        new_qkv.bias.data.copy_(old_qkv.bias.data[qkv_rows])

    # proj: [embed_dim, nh*hd] → [embed_dim, nh*kd]  (Linear weight: [out, in])
    new_proj = nn.Linear(nh * kd, embed_dim, bias=old_proj.bias is not None)
    new_proj.weight.data.copy_(old_proj.weight.data[:, proj_cols])
    if old_proj.bias is not None:
        new_proj.bias.data.copy_(old_proj.bias.data)

    attn.qkv  = new_qkv
    attn.proj = new_proj
    attn.head_dim = kd


def extract_subnet(model, masks, P_i):
    """Extract a standalone pruned model for subnet P_i.

    Returns:
        pruned_model   : nn.Module  — UViT with permanently reduced layers
        per_block_dims : dict       — {blk_name: {'ffn_hidden': int, 'attn_head_dim': int}}

    The pruned model runs the "full" forward path (no subnet_cfg) and
    produces the same output as the original model with subnet_cfg for P_i.
    """
    pruned = copy.deepcopy(model)
    pruned.cpu()

    blk_masks  = masks['masks'][P_i]
    ffn_ranks  = masks['ffn_internal_ranks']
    attn_ranks = masks['attn_channel_ranks']

    per_block_dims = {}

    for blk_name, blk_keep in blk_masks.items():
        block = _get_block(pruned, blk_name)
        dims  = {}

        # FFN pipe
        if 'ffn_keep' in blk_keep:
            k = blk_keep['ffn_keep']
            keep_idx = ffn_ranks[blk_name][:k]
            _prune_ffn(block.mlp, keep_idx)
            dims['ffn_hidden'] = k

        # Attention pipe
        if 'attn_keep' in blk_keep:
            kd = blk_keep['attn_keep']
            keep_idx = attn_ranks[blk_name][:kd]
            _prune_attn(block.attn, keep_idx)
            dims['attn_head_dim'] = kd

        per_block_dims[blk_name] = dims

    return pruned, per_block_dims


def reshape_to_pruned(model, per_block_dims):
    """Reshape a standard UViT to match pruned architecture dimensions.

    Creates new nn.Linear modules with target sizes but does NOT copy weights.
    Call model.load_state_dict() afterwards to populate the weights.

    This is used by the fine-tuning script to recreate the pruned architecture
    before loading the extracted state_dict.
    """
    for blk_name, dims in per_block_dims.items():
        block = _get_block(model, blk_name)

        if 'ffn_hidden' in dims:
            k = dims['ffn_hidden']
            in_f  = block.mlp.fc1.in_features
            out_f = block.mlp.fc2.out_features
            block.mlp.fc1 = nn.Linear(in_f, k, bias=block.mlp.fc1.bias is not None)
            block.mlp.fc2 = nn.Linear(k, out_f, bias=block.mlp.fc2.bias is not None)

        if 'attn_head_dim' in dims:
            kd = dims['attn_head_dim']
            nh = block.attn.num_heads
            embed_dim = block.attn.proj.out_features
            has_qkv_bias  = block.attn.qkv.bias is not None
            has_proj_bias = block.attn.proj.bias is not None
            block.attn.qkv  = nn.Linear(embed_dim, 3 * nh * kd, bias=has_qkv_bias)
            block.attn.proj = nn.Linear(nh * kd, embed_dim, bias=has_proj_bias)
            block.attn.head_dim = kd

    return model


# ===========================================================================
# CLI
# ===========================================================================

def _load_config(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('_cfg', path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def main():
    ap = argparse.ArgumentParser(
        description='Extract standalone pruned U-ViT subnets from OFA-trained model')
    ap.add_argument('--config', required=True,
                    help='ml_collections config .py file')
    ap.add_argument('--ckpt', required=True,
                    help='OFA-trained checkpoint directory (with nnet_ema.pth) '
                         'or single .pth file')
    ap.add_argument('--masks', required=True,
                    help='Path to uvit_masks.pt from uvit_prune.py')
    ap.add_argument('--p_values', required=True,
                    help='Comma-separated P_i values to extract, e.g. "0.25,0.5,0.75"')
    ap.add_argument('--outdir', default='outputs/extracted_subnets',
                    help='Output directory')
    ap.add_argument('--dataset_path', default=None,
                    help='Override dataset path (unused, for pipeline compat)')
    args = ap.parse_args()

    config = _load_config(args.config)

    # ── Build model and load OFA-trained weights ─────────────────────────────
    print(f'Building model from {args.config} …')
    model = utils.get_nnet(**config.nnet)

    if os.path.isfile(args.ckpt):
        ckpt_file = args.ckpt
    else:
        ckpt_file = os.path.join(args.ckpt, 'nnet_ema.pth')
        if not os.path.isfile(ckpt_file):
            ckpt_file = os.path.join(args.ckpt, 'nnet.pth')
    print(f'Loading OFA-trained weights from {ckpt_file}')

    sd = torch.load(ckpt_file, map_location='cpu', weights_only=True)
    if any(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    n_params_full = sum(p.numel() for p in model.parameters())
    print(f'  Full model: {n_params_full/1e6:.2f}M params')

    # ── Load masks ───────────────────────────────────────────────────────────
    masks = torch.load(args.masks, map_location='cpu', weights_only=False)
    print(f'  Masks: {len(masks["P_values"])} subnets')

    # ── Parse target P values ────────────────────────────────────────────────
    target_ps = [float(p) for p in args.p_values.split(',')]
    available = masks['P_values']

    os.makedirs(args.outdir, exist_ok=True)

    # ── Extract each subnet ──────────────────────────────────────────────────
    print(f'\n{"P_i":>8}  {"Params":>12}  {"Ratio":>8}  {"Path"}')
    print('-' * 60)

    for p_target in target_ps:
        # Find nearest P_i in masks
        P_i = min(available, key=lambda x: abs(float(x) - p_target))

        pruned, per_block_dims = extract_subnet(model, masks, P_i)
        n_params = sum(p.numel() for p in pruned.parameters())
        ratio = n_params / n_params_full

        p_str = f'{float(P_i):.4f}'.replace('.', 'p')
        save_dir = os.path.join(args.outdir, f'subnet_{p_str}')
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, 'model.pth')
        torch.save({
            'state_dict'     : pruned.state_dict(),
            'P_i'            : float(P_i),
            'per_block_dims' : per_block_dims,
            'n_params'       : n_params,
            'n_params_full'  : n_params_full,
            'config_nnet'    : dict(config.nnet),
        }, save_path)

        print(f'{float(P_i):>8.4f}  {n_params/1e6:>10.2f}M  {ratio:>7.1%}  {save_path}')

        # Verify forward pass works
        with torch.no_grad():
            x = torch.randn(1, 3, config.nnet.img_size, config.nnet.img_size)
            t = torch.tensor([500.0])
            n_classes = getattr(config.nnet, 'num_classes', 0)
            kwargs = {'y': torch.zeros(1, dtype=torch.long)} if n_classes > 0 else {}
            out = pruned(x, t, **kwargs)
            assert out.shape == x.shape, f'Shape mismatch: {out.shape} != {x.shape}'

    print(f'\n✅ Extracted {len(target_ps)} subnets → {args.outdir}')


if __name__ == '__main__':
    main()
