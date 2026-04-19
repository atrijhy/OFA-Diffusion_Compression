#!/usr/bin/env python3
"""Extract standalone pruned EDM subnets from an OFA-trained model.

After OFA joint training with physical slicing (ofa_train_edm_physical.py),
all subnets share the same full-size weight tensors and are selected at runtime
via conv_keep_idx / qkv_keep_idx.  This script permanently removes the pruned
channels, producing a standalone smaller model for each target P_i:

  - Conv pipe:  conv0 output rows + affine rows + norm1 channels + conv1 input
                cols are physically removed  →  smaller internal width
  - Attn pipe:  qkv output rows + proj input cols are physically removed
                →  smaller per-head dimension  (per-head-offset, all heads
                keep same offsets)

The extracted model runs the "full" forward path (no subnet_cfg needed)
with reduced internal dimensions — ready for independent fine-tuning.

Usage:
  cd /wherever/OFA/Diff-Pruning
  PYTHONPATH=/wherever/OFA/edm:$PYTHONPATH python edm_extract_subnet.py \
      --network   outputs/ofa_trained_physical/.../network-snapshot-XXXXXX.pkl \
      --masks     outputs/ofa_masks_physical/ofa_masks_physical.pt \
      --p_values  0.25,0.5,0.75 \
      --outdir    outputs/extracted_subnets_edm
"""

import os
import sys
import copy
import pickle
import click
import torch
import torch.nn as nn

# Ensure EDM & Diff-Pruning are on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EDM_REPO = os.path.join(SCRIPT_DIR, '..', 'edm')
sys.path.insert(0, os.path.abspath(EDM_REPO))
sys.path.insert(0, SCRIPT_DIR)

import dnnlib
from torch_utils import misc

# Import OFA network classes (needed for pickle unpickling)
import networks_ofa  # noqa: F401
from networks_ofa import (
    Conv2d, Linear, GroupNorm, SliceUNetBlock, SliceSongUNet,
    SliceVPPrecond, SliceVEPrecond, SliceEDMPrecond,
    UNetBlock,
)


# ===========================================================================
# Mask loading  (same as ofa_train_edm_physical.py)
# ===========================================================================

def _load_masks(masks_path):
    raw = torch.load(masks_path, map_location='cpu', weights_only=False)
    all_masks = raw['masks']
    conv_ranks = raw.get('conv_internal_ranks', {})
    qkv_ranks = raw.get('qkv_channel_ranks', {})
    P_values = sorted(all_masks.keys())
    return all_masks, conv_ranks, qkv_ranks, P_values


def _build_subnet_cfg(all_masks, P_i, conv_ranks, qkv_ranks):
    """Build subnet_cfg on CPU for extraction."""
    blk_masks = all_masks[P_i]
    subnet_cfg = {}
    for blk_name, blk_cfg in blk_masks.items():
        cfg = {}
        if 'conv_keep' in blk_cfg and blk_name in conv_ranks:
            k = blk_cfg['conv_keep']
            cfg['conv_keep_idx'] = conv_ranks[blk_name][:k]
        if 'qkv_keep' in blk_cfg and blk_name in qkv_ranks:
            kq = blk_cfg['qkv_keep']
            cfg['qkv_keep_idx'] = qkv_ranks[blk_name][:kq]
        if cfg:
            key = blk_name
            if key.startswith('model.'):
                key = key[len('model.'):]
            subnet_cfg[key] = cfg
    return subnet_cfg


# ===========================================================================
# Physical pruning of SliceUNetBlock → standard UNetBlock
# ===========================================================================

def _make_conv2d(ref, out_channels, in_channels, weight_data, bias_data):
    """Create a new Conv2d module with the same kernel/up/down/resample config
    as `ref`, but with different channel sizes and given weight/bias data."""
    kernel = ref.weight.shape[-1] if ref.weight is not None else 0
    new = Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel=kernel,
        bias=bias_data is not None,
        up=ref.up,
        down=ref.down,
        resample_filter=[1, 1],  # placeholder — we copy the buffer directly
        fused_resample=ref.fused_resample,
    )
    if weight_data is not None:
        new.weight = nn.Parameter(weight_data.clone())
    if bias_data is not None:
        new.bias = nn.Parameter(bias_data.clone())
    # Copy resample_filter buffer from the reference
    if ref.resample_filter is not None:
        new.resample_filter = ref.resample_filter.clone()
    else:
        new.resample_filter = None
    return new


def _prune_conv_pipe(block, conv_keep_idx):
    """Physically prune the conv pipe of a SliceUNetBlock.

    Conv pipe:
      conv0:  [out_ch, in_ch, k, k] → keep rows at conv_keep_idx
      affine: [out_ch*(2 if adaptive_scale else 1), emb_ch] → keep corresponding rows
      norm1:  [out_ch] weight/bias → keep at conv_keep_idx
      conv1:  [out_ch, out_ch, k, k] → keep cols at conv_keep_idx (input dim)
    """
    keep = conv_keep_idx
    ck = keep.numel()
    out_ch = block.out_channels

    # ── conv0: keep output rows ──────────────────────────────────────────────
    old_conv0 = block.conv0
    w0 = old_conv0.weight[keep]                           # [ck, in_ch, k, k]
    b0 = old_conv0.bias[keep] if old_conv0.bias is not None else None
    block.conv0 = _make_conv2d(old_conv0, ck, old_conv0.in_channels, w0, b0)

    # ── affine: keep corresponding rows ──────────────────────────────────────
    old_affine = block.affine
    if block.adaptive_scale:
        # Affine output is [2*out_ch] = [scale, shift], each of out_ch
        aff_idx = torch.cat([keep, keep + out_ch])      # [2*ck]
    else:
        aff_idx = keep                                    # [ck]
    new_out = aff_idx.numel()
    new_affine = Linear(
        in_features=old_affine.in_features,
        out_features=new_out,
        bias=old_affine.bias is not None,
    )
    new_affine.weight = nn.Parameter(old_affine.weight[aff_idx].clone())
    if old_affine.bias is not None:
        new_affine.bias = nn.Parameter(old_affine.bias[aff_idx].clone())
    block.affine = new_affine

    # ── norm1: keep channels at conv_keep_idx ────────────────────────────────
    old_norm1 = block.norm1
    new_ng = min(old_norm1.num_groups, ck)
    while ck % new_ng != 0 and new_ng > 1:
        new_ng -= 1
    new_norm1 = GroupNorm(num_channels=ck, num_groups=new_ng, eps=old_norm1.eps)
    new_norm1.weight = nn.Parameter(old_norm1.weight[keep].clone())
    new_norm1.bias = nn.Parameter(old_norm1.bias[keep].clone())
    block.norm1 = new_norm1

    # ── conv1: keep input cols ───────────────────────────────────────────────
    old_conv1 = block.conv1
    w1 = old_conv1.weight[:, keep]                        # [out_ch, ck, k, k]
    b1 = old_conv1.bias.clone() if old_conv1.bias is not None else None
    block.conv1 = _make_conv2d(old_conv1, out_ch, ck, w1, b1)

    return ck


def _prune_attn_pipe(block, qkv_keep_idx):
    """Physically prune the attention pipe of a SliceUNetBlock.

    Attention pipe (per-head-offset, uniform across all heads):
      qkv:  [3*out_ch, out_ch, 1, 1] → keep rows for Q,K,V per head
      proj: [out_ch, out_ch, 1, 1]   → keep cols per head
    """
    keep = qkv_keep_idx
    kd = keep.numel()
    nh = block.num_heads
    hd = block.head_dim
    out_ch = block.out_channels

    # ── qkv: build absolute row indices ──────────────────────────────────────
    qkv_rows = []
    for h in range(nh):
        base_q = h * hd
        base_k = nh * hd + h * hd
        base_v = 2 * nh * hd + h * hd
        qkv_rows.append(base_q + keep)
        qkv_rows.append(base_k + keep)
        qkv_rows.append(base_v + keep)
    qkv_rows = torch.cat(qkv_rows)  # [3*nh*kd]

    old_qkv = block.qkv
    w_qkv = old_qkv.weight[qkv_rows]                     # [3*nh*kd, out_ch, 1, 1]
    b_qkv = old_qkv.bias[qkv_rows] if old_qkv.bias is not None else None
    new_qkv = Conv2d(
        in_channels=out_ch, out_channels=3 * nh * kd, kernel=1,
        bias=b_qkv is not None,
    )
    new_qkv.weight = nn.Parameter(w_qkv.clone())
    if b_qkv is not None:
        new_qkv.bias = nn.Parameter(b_qkv.clone())
    block.qkv = new_qkv

    # ── proj: keep input cols per head ───────────────────────────────────────
    proj_cols = []
    for h in range(nh):
        proj_cols.append(h * hd + keep)
    proj_cols = torch.cat(proj_cols)  # [nh*kd]

    old_proj = block.proj
    w_proj = old_proj.weight[:, proj_cols]                 # [out_ch, nh*kd, 1, 1]
    b_proj = old_proj.bias.clone() if old_proj.bias is not None else None
    new_proj = Conv2d(
        in_channels=nh * kd, out_channels=out_ch, kernel=1,
        bias=b_proj is not None,
    )
    new_proj.weight = nn.Parameter(w_proj.clone())
    if b_proj is not None:
        new_proj.bias = nn.Parameter(b_proj.clone())
    block.proj = new_proj

    # Update head_dim on the block
    block.head_dim = kd

    return kd


# ===========================================================================
# Subnet extraction (top level)
# ===========================================================================

def extract_edm_subnet(net, all_masks, conv_ranks, qkv_ranks, P_i):
    """Extract a standalone pruned EDM model for subnet P_i.

    net: SliceVPPrecond / SliceVEPrecond / SliceEDMPrecond (EMA model)

    Returns:
        pruned_net:    nn.Module — preconditioner with permanently reduced model
        per_block_dims: dict   — {blk_name: {'conv_internal': int, 'attn_head_dim': int}}
    """
    pruned = copy.deepcopy(net)
    pruned.cpu()

    subnet_cfg = _build_subnet_cfg(all_masks, P_i, conv_ranks, qkv_ranks)

    per_block_dims = {}

    # Iterate over all SliceUNetBlock modules inside the model
    model = pruned.model  # SliceSongUNet

    for part_name in ['enc', 'dec']:
        part = getattr(model, part_name)
        for blk_name, block in part.items():
            if not isinstance(block, SliceUNetBlock):
                continue

            full_key = f'{part_name}.{blk_name}'
            dims = {}

            blk_cfg = subnet_cfg.get(full_key, {})

            # Conv pipe
            conv_keep_idx = blk_cfg.get('conv_keep_idx', None)
            if conv_keep_idx is not None:
                ck = _prune_conv_pipe(block, conv_keep_idx)
                dims['conv_internal'] = ck
            else:
                dims['conv_internal'] = block.out_channels

            # Attn pipe
            qkv_keep_idx = blk_cfg.get('qkv_keep_idx', None)
            if qkv_keep_idx is not None and block.num_heads > 0:
                kd = _prune_attn_pipe(block, qkv_keep_idx)
                dims['attn_head_dim'] = kd
            elif block.num_heads > 0:
                dims['attn_head_dim'] = block.head_dim

            per_block_dims[full_key] = dims

    return pruned, per_block_dims


def reshape_edm_to_pruned(net, per_block_dims):
    """Reshape a standard EDM model to match pruned architecture dimensions.

    Creates new Conv2d/Linear/GroupNorm with target sizes but does NOT copy
    weights.  Call net.load_state_dict() afterwards to populate weights.

    This is used by the fine-tuning script to recreate the pruned architecture
    before loading the extracted state_dict.
    """
    model = net.model  # SliceSongUNet

    for full_key, dims in per_block_dims.items():
        part_name, blk_name = full_key.split('.', 1)
        part = getattr(model, part_name)
        block = part[blk_name]

        if not isinstance(block, SliceUNetBlock):
            continue

        out_ch = block.out_channels

        # ── Conv pipe ────────────────────────────────────────────────────────
        ck = dims.get('conv_internal', out_ch)
        if ck != out_ch:
            # conv0: output channels → ck
            old_conv0 = block.conv0
            block.conv0 = _make_conv2d(
                old_conv0, ck, old_conv0.in_channels,
                torch.zeros(ck, old_conv0.in_channels, old_conv0.weight.shape[2], old_conv0.weight.shape[3]),
                torch.zeros(ck) if old_conv0.bias is not None else None,
            )
            if old_conv0.resample_filter is not None:
                block.conv0.resample_filter = old_conv0.resample_filter.clone()

            # affine
            old_aff = block.affine
            aff_out = ck * (2 if block.adaptive_scale else 1)
            block.affine = Linear(old_aff.in_features, aff_out, bias=old_aff.bias is not None)

            # norm1
            old_norm1 = block.norm1
            ng = min(old_norm1.num_groups, ck)
            while ck % ng != 0 and ng > 1:
                ng -= 1
            block.norm1 = GroupNorm(num_channels=ck, num_groups=ng, eps=old_norm1.eps)

            # conv1: input channels → ck
            old_conv1 = block.conv1
            block.conv1 = _make_conv2d(
                old_conv1, out_ch, ck,
                torch.zeros(out_ch, ck, old_conv1.weight.shape[2], old_conv1.weight.shape[3]),
                torch.zeros(out_ch) if old_conv1.bias is not None else None,
            )

        # ── Attn pipe ────────────────────────────────────────────────────────
        kd = dims.get('attn_head_dim', None)
        if kd is not None and block.num_heads > 0:
            nh = block.num_heads
            old_hd = block.head_dim

            if kd != old_hd:
                # qkv
                block.qkv = Conv2d(
                    in_channels=out_ch, out_channels=3 * nh * kd, kernel=1,
                    bias=block.qkv.bias is not None,
                )
                # proj
                block.proj = Conv2d(
                    in_channels=nh * kd, out_channels=out_ch, kernel=1,
                    bias=block.proj.bias is not None,
                )
                block.head_dim = kd

    return net


# ===========================================================================
# CLI
# ===========================================================================

@click.command()
@click.option('--network', 'network_pkl', help='OFA network-snapshot-*.pkl',
              metavar='PKL', type=str, required=True)
@click.option('--masks', 'masks_path', help='Path to ofa_masks_physical.pt',
              metavar='PT', type=str, required=True)
@click.option('--p_values', help='Comma-separated P_i values to extract',
              metavar='LIST', type=str, required=True)
@click.option('--outdir', help='Output directory',
              metavar='DIR', type=str, default='outputs/extracted_subnets_edm')
def main(network_pkl, masks_path, p_values, outdir):
    """Extract standalone pruned subnets from an OFA-trained EDM model.

    Each subnet is saved as an EDM pickle (.pkl) in the same format as
    network-snapshot-*.pkl, ready for generation or fine-tuning.
    """
    # ── Load OFA network ─────────────────────────────────────────────────────
    print(f'Loading OFA network from "{network_pkl}" …')
    with dnnlib.util.open_url(network_pkl, verbose=True) as f:
        data = pickle.load(f)
    net = data['ema']
    net.eval()

    n_params_full = sum(p.numel() for p in net.parameters())
    print(f'  Full model: {n_params_full/1e6:.2f}M params')

    # ── Load masks ───────────────────────────────────────────────────────────
    print(f'Loading masks from "{masks_path}" …')
    all_masks, conv_ranks, qkv_ranks, P_available = _load_masks(masks_path)
    print(f'  {len(P_available)} subnets available')

    # ── Parse target P values ────────────────────────────────────────────────
    target_ps = [float(p) for p in p_values.split(',')]

    os.makedirs(outdir, exist_ok=True)

    # ── Extract each subnet ──────────────────────────────────────────────────
    print(f'\n{"P_i":>8}  {"Params":>12}  {"Ratio":>8}  {"Path"}')
    print('-' * 60)

    for p_target in target_ps:
        P_i = min(P_available, key=lambda x: abs(float(x) - p_target))

        pruned, per_block_dims = extract_edm_subnet(
            net, all_masks, conv_ranks, qkv_ranks, P_i)
        n_params = sum(p.numel() for p in pruned.parameters())
        ratio = n_params / n_params_full

        p_str = f'{float(P_i):.4f}'.replace('.', 'p')
        save_dir = os.path.join(outdir, f'subnet_{p_str}')
        os.makedirs(save_dir, exist_ok=True)

        # Save as EDM pickle (same format as network-snapshot-*.pkl)
        save_path = os.path.join(save_dir, 'network-snapshot.pkl')
        save_data = {
            'ema': pruned,
            'loss_fn': data.get('loss_fn', None),
            'augment_pipe': data.get('augment_pipe', None),
            'dataset_kwargs': data.get('dataset_kwargs', None),
            # Extra metadata for fine-tuning
            'P_i': float(P_i),
            'per_block_dims': per_block_dims,
            'n_params': n_params,
            'n_params_full': n_params_full,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f'{float(P_i):>8.4f}  {n_params/1e6:>10.2f}M  {ratio:>7.1%}  {save_path}')

        # Verify forward pass works
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pruned.to(device).eval()
        with torch.no_grad():
            x = torch.randn(1, pruned.img_channels, pruned.img_resolution,
                             pruned.img_resolution, device=device)
            sigma = torch.ones(1, device=device)
            out = pruned(x, sigma)
            assert out.shape == x.shape, f'Shape mismatch: {out.shape} != {x.shape}'
        pruned.cpu()

    print(f'\n✅ Extracted {len(target_ps)} subnets → {outdir}')


if __name__ == '__main__':
    main()
