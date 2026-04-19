# prune.py  —  U-ViT OFA pruning with physical-slicing-aware importance
#
# Adapted from ddpm_prune_physical.py (EDM) / sd_prune_physical.py (SD) for U-ViT.
#
# Key differences from EDM / SD versions:
#   - U-ViT is a pure ViT with U-shaped skip connections (no multi-scale convolutions).
#   - All blocks share the same embed_dim — no multi-resolution feature maps.
#   - QKV is a merged linear layer [3*embed_dim, embed_dim] (like EDM, unlike SD's separate Q/K/V).
#   - No Conv pipe — PatchEmbed and final_layer are I/O layers, not internal width.
#   - Two prunable pipes per block: FFN pipe (fc1↔fc2) + Attention pipe (qkv↔proj).
#
# Prunable pipes per block (= one transformer block):
#
#   1. FFN pipe:   fc1 (output) + fc2 (input)
#      Per-neuron importance aggregated from both ends.
#      Ranks shape: [hidden_dim]   (e.g. 2048 for mlp_ratio=4, embed_dim=512)
#
#   2. Attn pipe:  qkv (output) + proj (input)
#      Per-head-offset aggregation — all heads keep the same offsets.
#      This ensures valid multi-head attention regardless of num_heads.
#      Ranks shape: [head_dim]     (e.g. 64 for embed_dim=512, num_heads=8)
#      Actual kept attention width = nh * attn_keep
#
# Resolution groups (Formula 5 block partition):
#   U-ViT is a pure ViT with no spatial downsampling, but has U-shaped skip
#   connections dividing blocks into in_blocks / mid_block / out_blocks.
#   We treat these 3 groups as the "resolution groups":
#
#     in_blocks  : in_blocks.{0, ..., depth//2-1}    K_in  = depth//2 blocks
#     mid_block  : mid_block                          K_mid = 1 block
#     out_blocks : out_blocks.{0, ..., depth//2-1}   K_out = depth//2 blocks
#
#   Grouping modes (--grouping_mode):
#
#   stage (default):
#     3 groups (in/mid/out). Formula 5 allocates C_l across blocks within
#     each group. Both FFN and Attn scale from the same shared C_l.
#
#   per_block:
#     Each block is its own group → C_l = P*embed_dim uniformly for all blocks.
#     Equivalent to standard uniform pruning.
#
#   global:
#     All blocks in one group. Formula 5 allocates C_l across ALL blocks.
#
#   per_block_perpipe  ← new:
#     Each block independently allocates its total budget P*(hidden_dim+embed_dim)
#     between the FFN pipe and the Attn pipe in proportion to their relative
#     per-channel average importance. A block where attention is more important
#     than FFN keeps more attention head_dim offsets and fewer FFN neurons.
#     Unlike all other modes, FFN and Attn are NOT tied to the same C_l.
#
#   (Legacy per-pipe mode is kept as commented code in build_masks().)
#
# Mask format (consumed by build_subnet_cfgs in uvit_train_ofa_physical.py):
#   {
#     'masks': {
#       P_i: {
#         'in_blocks.0':  {'ffn_keep': 1536, 'attn_keep': 48},
#         'mid_block':    {'ffn_keep': 2048, 'attn_keep': 64},
#         ...
#       }
#     },
#     'ffn_internal_ranks':   {blk_name: LongTensor[hidden_dim] sorted desc},
#     'attn_channel_ranks':   {blk_name: LongTensor[head_dim]  sorted desc},
#     'P_values':   [0.25, 0.275, ..., 1.0],
#     'num_heads':  int,
#     'head_dim':   int,
#     'hidden_dim': int,
#     'embed_dim':  int,
#   }
#
# Usage:
#   python uvit_prune.py \
#       --config          configs/cifar10_uvit_small.py \
#       --ckpt            <pretrained .pth or ckpt dir> \
#       --dataset_path    <cifar10 data dir> \
#       --outdir          outputs/uvit_masks \
#       --n_importance_samples 1024 \
#       --batch_size      64 \
#       --device          cuda

import argparse
import importlib.util
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config loader (avoids absl/flags dependency)
# ---------------------------------------------------------------------------

def load_config(path: str):
    spec = importlib.util.spec_from_file_location("_uvit_cfg", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


# ---------------------------------------------------------------------------
# Per-channel Taylor importance  (Formula 4, real data, VP-SDE LSimple)
# ---------------------------------------------------------------------------

def compute_taylor_importance(model, config, dataset_path, n_importance_samples,
                               batch_size, device):
    """
    Per-channel Taylor importance (paper Formula 4):
        I_i^C = (1/N) * Σ_n |c_i^T ∇L_n|

    For FFN:
      fc1 output importance:  (fc1.weight * grad).sum(dim=1).abs()  → [hidden_dim]
      fc2 input importance:   (fc2.weight * grad).sum(dim=0).abs()  → [hidden_dim]
      Pipe importance = fc1_output_imp + fc2_input_imp

    For Attention:
      qkv output importance:  (qkv.weight * grad).sum(dim=1).abs()  → [3*nh*hd]
      proj input importance:  (proj.weight * grad).sum(dim=0).abs()  → [nh*hd]
      Per-head-offset aggregation:
        I[j] = Σ_h ( I_Q[h*hd+j] + I_K[nh*hd+h*hd+j] + I_V[2*nh*hd+h*hd+j] )
             + Σ_h I_proj_in[h*hd+j]

    Returns:
        ffn_imp       : {blk_name: tensor[hidden_dim]}   per-neuron pipe importance
        attn_imp      : {blk_name: tensor[head_dim]}     per-head-offset pipe importance
        num_heads     : int
        head_dim      : int
        hidden_dim    : int
    """
    import sde as sde_module
    from datasets import get_dataset

    # ── Dataset ──────────────────────────────────────────────────────────────
    cfg_dataset = dict(config.dataset)
    cfg_dataset['path'] = dataset_path
    dataset_obj = get_dataset(**cfg_dataset)
    is_cond     = (config.train.mode == 'cond')
    train_set   = dataset_obj.get_split(split='train', labeled=is_cond)
    loader      = DataLoader(train_set, batch_size=batch_size,
                             shuffle=True, drop_last=True,
                             num_workers=4, pin_memory=True)
    loader_iter = iter(loader)

    # ── Score model (VP-SDE, same as training) ────────────────────────────────
    score_model = sde_module.ScoreModel(model, pred=config.pred, sde=sde_module.VPSDE())

    # ── Identify target layers ────────────────────────────────────────────────
    # Each transformer block has:
    #   block.mlp.fc1  (Linear [embed_dim → hidden_dim])
    #   block.mlp.fc2  (Linear [hidden_dim → embed_dim])
    #   block.attn.qkv (Linear [embed_dim → 3*embed_dim])
    #   block.attn.proj(Linear [embed_dim → embed_dim])

    ffn_fc1    = {}   # {blk_name: nn.Linear}
    ffn_fc2    = {}   # {blk_name: nn.Linear}
    attn_mods  = {}   # {blk_name: (attn_module, nh, hd)}
    num_heads  = None
    head_dim   = None
    hidden_dim = None

    for name, m in model.named_modules():
        if name.endswith('.mlp.fc1') and isinstance(m, nn.Linear):
            blk_name = name[: -len('.mlp.fc1')]
            ffn_fc1[blk_name] = m
            hidden_dim = m.out_features

        elif name.endswith('.mlp.fc2') and isinstance(m, nn.Linear):
            blk_name = name[: -len('.mlp.fc2')]
            ffn_fc2[blk_name] = m

        elif (name.endswith('.attn')
              and hasattr(m, 'num_heads')
              and hasattr(m, 'qkv')
              and isinstance(m.qkv, nn.Linear)):
            blk_name = name[: -len('.attn')]
            nh = m.num_heads
            hd = m.qkv.out_features // (3 * nh)
            attn_mods[blk_name] = (m, nh, hd)
            num_heads = nh
            head_dim  = hd

    if not ffn_fc1 and not attn_mods:
        raise RuntimeError(
            'No prunable layers found. Expected modules named ".mlp.fc1" '
            'and ".attn" (with .qkv child). Check model module names.')

    print(f'  Found {len(ffn_fc1)} FFN pipes, {len(attn_mods)} Attn pipes')
    print(f'  num_heads={num_heads}, head_dim={head_dim}, hidden_dim={hidden_dim}')

    # ── Importance accumulators ──────────────────────────────────────────────
    # FFN pipe: fc1 output + fc2 input
    fc1_out_imp = {b: torch.zeros(hidden_dim) for b in ffn_fc1}
    fc2_in_imp  = {b: torch.zeros(hidden_dim) for b in ffn_fc2}

    # Attn pipe: qkv output + proj input
    embed_dim = num_heads * head_dim
    qkv_out_imp  = {b: torch.zeros(3 * embed_dim) for b in attn_mods}
    proj_in_imp  = {b: torch.zeros(embed_dim)     for b in attn_mods}

    # ── Per-sample importance loop (Formula 4, no cross-sample cancellation) ──
    model.requires_grad_(True)
    model.eval()   # eval mode: disables dropout for deterministic importance scoring

    print(f'  Accumulating per-channel Taylor importance '
          f'({n_importance_samples} individual backward passes) …')

    for _ in tqdm(range(n_importance_samples), desc='importance samples'):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        if isinstance(batch, (list, tuple)):
            x = batch[0][:1].to(device)
            y = batch[1][:1].to(device)
        else:
            x = batch[:1].to(device)
            y = None
        # 1 sample → no cross-sample gradient cancellation

        model.zero_grad()
        y_kwargs = dict(y=y) if y is not None else {}
        loss = sde_module.LSimple(score_model, x, pred=config.pred, **y_kwargs)
        loss.mean().backward()

        with torch.no_grad():
            # FFN fc1: per output-neuron importance
            for blk_name, m in ffn_fc1.items():
                if m.weight.grad is None:
                    continue
                fc1_out_imp[blk_name] += (
                    m.weight.detach() * m.weight.grad
                ).sum(dim=1).abs().cpu()   # [hidden_dim]

            # FFN fc2: per input-neuron importance (back end of FFN pipe)
            for blk_name, m in ffn_fc2.items():
                if m.weight.grad is None:
                    continue
                fc2_in_imp[blk_name] += (
                    m.weight.detach() * m.weight.grad
                ).sum(dim=0).abs().cpu()   # [hidden_dim]

            # Attention qkv: per output-channel importance
            for blk_name, (attn_m, nh, hd) in attn_mods.items():
                if attn_m.qkv.weight.grad is None:
                    continue
                qkv_out_imp[blk_name] += (
                    attn_m.qkv.weight.detach() * attn_m.qkv.weight.grad
                ).sum(dim=1).abs().cpu()   # [3*nh*hd]

                # proj: per input-channel importance (back end of attention pipe)
                if attn_m.proj.weight.grad is not None:
                    proj_in_imp[blk_name] += (
                        attn_m.proj.weight.detach() * attn_m.proj.weight.grad
                    ).sum(dim=0).abs().cpu()   # [nh*hd]

    model.eval()
    model.requires_grad_(False)

    # Average over N samples
    for b in fc1_out_imp:
        fc1_out_imp[b] /= n_importance_samples
    for b in fc2_in_imp:
        fc2_in_imp[b] /= n_importance_samples
    for b in qkv_out_imp:
        qkv_out_imp[b] /= n_importance_samples
    for b in proj_in_imp:
        proj_in_imp[b] /= n_importance_samples

    # ── Aggregate FFN pipe importance ─────────────────────────────────────────
    ffn_imp = {}
    for blk_name in ffn_fc1:
        imp_front = fc1_out_imp.get(blk_name, torch.zeros(hidden_dim))
        imp_back  = fc2_in_imp.get(blk_name, torch.zeros(hidden_dim))
        ffn_imp[blk_name] = imp_front + imp_back   # [hidden_dim]

    # ── Aggregate Attn pipe importance (per-head-offset) ──────────────────────
    # qkv layout: [Q_h0..Q_h(nh-1), K_h0..K_h(nh-1), V_h0..V_h(nh-1)]
    # For each within-head offset j ∈ [0, hd), aggregate importance across
    # all heads and Q/K/V sections + proj input.
    attn_imp = {}
    for blk_name, (_, nh, hd) in attn_mods.items():
        qkv_imp = qkv_out_imp[blk_name]   # [3*nh*hd]

        offset_imp = torch.zeros(hd)
        for h in range(nh):
            q_base = h * hd
            k_base = nh * hd + h * hd
            v_base = 2 * nh * hd + h * hd
            offset_imp += (qkv_imp[q_base:q_base+hd]
                          + qkv_imp[k_base:k_base+hd]
                          + qkv_imp[v_base:v_base+hd])

        # Add proj input importance (back end of attention pipe)
        p_imp = proj_in_imp[blk_name]   # [nh*hd]
        for h in range(nh):
            offset_imp += p_imp[h*hd:(h+1)*hd]

        attn_imp[blk_name] = offset_imp   # [hd]

    return ffn_imp, attn_imp, num_heads, head_dim, hidden_dim


# ---------------------------------------------------------------------------
# Mask builder  (Algorithm 1 + Formula 5)
# ---------------------------------------------------------------------------

def build_masks(ffn_imp, attn_imp, P_values, num_heads, head_dim, hidden_dim,
                p_min=0.25, grouping_mode='stage'):
    """
    Build nested keep-masks for each P_i (Algorithm 1 + Formula 5).

    Grouping modes for Formula 5:
      stage            : in_blocks / mid_block / out_blocks (default).
      per_block        : each block is its own group → uniform pruning.
      global           : all blocks share one group.
      per_block_perpipe: within each block, independently allocate budget to
                         FFN and Attn based on relative per-channel importance.
                         Total block budget = P*(hidden_dim+embed_dim), split by
                         (mean_ffn_imp / (mean_ffn_imp + mean_attn_imp)) ratio.

    Block budget (shared-C_l mode):
      1) For each block l, aggregate FFN+Attn to a shared block score:
           I_l^L = I_l^(ffn) + I_l^(attn)
      2) Within each group B, allocate one shared channel budget C_l by
         Formula 5 on block width |l| = embed_dim:
           hat_I_l = I_l^L / Σ_{l'∈B} I_{l'}^L
           C_l = max{ min{ hat_I_l * K_B * P_i * |l|, |l| }, P_0 * |l| }
      3) Derive pipe keeps from the same C_l:
           FFN  : ffn_keep  = round((C_l / embed_dim) * hidden_dim)
           Attn : attn_keep = round(C_l / num_heads)   (per-head offsets)

    Importance ranking determines WHICH neurons/offsets to keep (argsort desc,
    prefix-truncation guarantees nestedness S_{P_i} ⊆ S_{P_{i+1}}).

    Returns
    -------
    ffn_ranks   : {blk_name: LongTensor[hidden_dim]}  sorted desc
    attn_ranks  : {blk_name: LongTensor[head_dim]}    sorted desc
    all_masks   : {P_i: {blk_name: {'ffn_keep': int, 'attn_keep': int}}}
    """
    ffn_ranks  = {b: imp.argsort(descending=True) for b, imp in ffn_imp.items()}
    attn_ranks = {b: imp.argsort(descending=True) for b, imp in attn_imp.items()}
    # Shared block-width axis used by Formula 5.
    embed_dim = num_heads * head_dim

    # ── Build allocation groups ───────────────────────────────────────────────
    # Collect all block names from both dicts
    all_blocks = sorted(set(list(ffn_imp.keys()) + list(attn_imp.keys())))

    res_groups = {}   # group_key -> [blk_name, ...]
    for blk_name in all_blocks:
        if grouping_mode in ('per_block', 'per_block_perpipe'):
            group_key = blk_name
        elif grouping_mode == 'global':
            group_key = 'global'
        elif grouping_mode == 'stage':
            if blk_name.startswith('in_blocks'):
                group_key = 'in_blocks'
            elif blk_name.startswith('mid_block'):
                group_key = 'mid_block'
            elif blk_name.startswith('out_blocks'):
                group_key = 'out_blocks'
            else:
                group_key = 'other'
        else:
            raise ValueError(f'Unknown grouping_mode: {grouping_mode}')
        res_groups.setdefault(group_key, []).append(blk_name)

    print(f'\n  Allocation groups (mode={grouping_mode}):')
    for gk, blk_names in res_groups.items():
        print(f'    {gk}: {len(blk_names)} blocks')

    # ── Build masks for each P_i ─────────────────────────────────────────────
    # For stage/per_block/global: shared-C_l Formula 5 on block axis |l|=embed_dim.
    # For per_block_perpipe: per-pipe Formula 5 inside each block (FFN/Attn as layers).
    all_masks = {}
    mode_desc = (
        'Formula 5 shared-C_l'
        if grouping_mode in ('stage', 'per_block', 'global')
        else 'Formula 5 per-pipe'
    )
    print(f'\n  Computing OFA masks (Algorithm 1, {mode_desc}, '
          f'grouping_mode={grouping_mode}):')

    for P in P_values:
        mask = {}

        # P >= 1.0: full model
        if P >= 1.0:
            for blk_name in all_blocks:
                blk_cfg = {}
                if blk_name in ffn_imp:
                    blk_cfg['ffn_keep'] = hidden_dim
                if blk_name in attn_imp:
                    blk_cfg['attn_keep'] = head_dim
                mask[blk_name] = blk_cfg
            all_masks[P] = mask
            print(f'    P_i={P:.4f}  avg_ffn=1.000  avg_attn=1.000')
            continue

        # P < 1.0: shared-C_l Formula 5 per allocation group
        for group_key, blk_names in res_groups.items():
            if not blk_names:
                continue

            # ── per_block_perpipe: independent FFN/Attn allocation per block ──
            if grouping_mode == 'per_block_perpipe':
                c_min_ffn = max(1, round(p_min * hidden_dim))
                c_min_hd  = max(1, round(p_min * head_dim))
                for bn in blk_names:   # always exactly 1 block per group here
                    blk_cfg  = mask.setdefault(bn, {})
                    has_ffn  = bn in ffn_imp
                    has_attn = bn in attn_imp

                    # Formula 5 layer score: I_l^L = sum_c I_c  (raw channel sum,
                    # consistent with all other grouping modes).
                    i_ffn  = ffn_imp[bn].sum().item()  if has_ffn  else 0.0
                    i_attn = attn_imp[bn].sum().item() if has_attn else 0.0
                    total_i = max(i_ffn + i_attn, 1e-12)

                    # Formula 5 with K_B = 2 (FFN and Attn are the two "layers"
                    # in the group).  Each pipe uses its own |l|:
                    #   C_ffn        = max(min(hat_I_ffn  * 2*P * hidden_dim, hidden_dim), P_0*hidden_dim)
                    #   C_attn_embed = max(min(hat_I_attn * 2*P * embed_dim,  embed_dim),  P_0*embed_dim)
                    K_B = 2

                    if has_ffn:
                        ffn_alloc = (i_ffn / total_i) * K_B * P * hidden_dim
                        ffn_keep  = max(c_min_ffn, min(round(ffn_alloc), hidden_dim))
                        blk_cfg['ffn_keep'] = ffn_keep

                    if has_attn:
                        attn_alloc_embed = (i_attn / total_i) * K_B * P * embed_dim
                        attn_keep = max(c_min_hd,
                                        min(round(attn_alloc_embed / num_heads), head_dim))
                        blk_cfg['attn_keep'] = attn_keep
                continue   # skip shared-C_l logic below

            # Build per-block shared importance I_l^L = I_l^(ffn) + I_l^(attn)
            block_imp = {}
            for bn in blk_names:
                imp = 0.0
                if bn in ffn_imp:
                    imp += ffn_imp[bn].sum().item()
                if bn in attn_imp:
                    imp += attn_imp[bn].sum().item()
                block_imp[bn] = imp

            K_group = len(blk_names)
            total_block_imp = max(sum(block_imp.values()), 1e-12)
            c_min = max(1, round(p_min * embed_dim))

            for bn in blk_names:
                norm_l = block_imp[bn] / total_block_imp
                C_raw = norm_l * K_group * P * embed_dim
                # Shared block budget C_l is defined on the block reference
                # width axis |l| = embed_dim (not on FFN hidden_dim/head_dim).
                C_l = max(c_min, min(round(C_raw), embed_dim))

                blk_cfg = mask.setdefault(bn, {})
                if bn in ffn_imp:
                    ffn_keep = max(1, round((C_l / embed_dim) * hidden_dim))
                    ffn_keep = min(ffn_keep, hidden_dim)
                    blk_cfg['ffn_keep'] = ffn_keep
                if bn in attn_imp:
                    # Map shared C_l (embed axis) to per-head offset axis.
                    # Equivalent to round(C_l / num_heads), but writing it as
                    # ratio*head_dim keeps the same unit-conversion style as FFN.
                    attn_keep = max(1, round((C_l / embed_dim) * head_dim))
                    attn_keep = min(attn_keep, head_dim)
                    blk_cfg['attn_keep'] = attn_keep

                    # Legacy equivalent form (kept for reference):
                    # attn_keep = max(1, round(C_l / num_heads))

            # --- Legacy per-pipe allocation (kept for reference, disabled) ---
            #
            # ffn_blks  = [bn for bn in blk_names if bn in ffn_imp]
            # attn_blks = [bn for bn in blk_names if bn in attn_imp]
            #
            # if ffn_blks:
            #     K_ffn = len(ffn_blks)
            #     total_ffn_imp = max(
            #         sum(ffn_imp[bn].sum().item() for bn in ffn_blks), 1e-12)
            #     c_min_ffn = max(1, round(p_min * hidden_dim))
            #     for bn in ffn_blks:
            #         I_l = ffn_imp[bn].sum().item()
            #         norm_l = I_l / total_ffn_imp
            #         C_raw = norm_l * K_ffn * P * hidden_dim
            #         ffn_keep = max(c_min_ffn, min(round(C_raw), hidden_dim))
            #         mask.setdefault(bn, {})['ffn_keep'] = ffn_keep
            #
            # if attn_blks:
            #     K_attn = len(attn_blks)
            #     total_attn_imp = max(
            #         sum(attn_imp[bn].sum().item() for bn in attn_blks), 1e-12)
            #     c_min_attn = max(1, round(p_min * head_dim))
            #     for bn in attn_blks:
            #         I_l = attn_imp[bn].sum().item()
            #         norm_l = I_l / total_attn_imp
            #         C_raw = norm_l * K_attn * P * head_dim
            #         attn_keep = max(c_min_attn, min(round(C_raw), head_dim))
            #         mask.setdefault(bn, {})['attn_keep'] = attn_keep

        all_masks[P] = mask

        # Stats
        ffn_ratios = []
        attn_ratios = []
        for bn in all_blocks:
            if bn in mask:
                cfg = mask[bn]
                if 'ffn_keep' in cfg:
                    ffn_ratios.append(cfg['ffn_keep'] / hidden_dim)
                if 'attn_keep' in cfg:
                    attn_ratios.append(cfg['attn_keep'] / head_dim)
        avg_ffn  = sum(ffn_ratios)  / len(ffn_ratios)  if ffn_ratios  else 1.0
        avg_attn = sum(attn_ratios) / len(attn_ratios) if attn_ratios else 1.0
        print(f'    P_i={P:.4f}  avg_ffn={avg_ffn:.3f}  avg_attn={avg_attn:.3f}')

    return ffn_ranks, attn_ranks, all_masks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='U-ViT OFA pruning (physical slicing)')
    ap.add_argument('--config',   required=True,
                    help='Path to ml_collections config .py file')
    ap.add_argument('--ckpt',     required=True,
                    help='Checkpoint: .pth file OR directory with nnet_ema.pth/nnet.pth')
    ap.add_argument('--dataset_path', default=None,
                    help='Override dataset path from config')
    ap.add_argument('--outdir',   default='outputs/uvit_masks',
                    help='Output directory for uvit_masks.pt')
    ap.add_argument('--device',   default='cuda')
    ap.add_argument('--p_min',    type=float, default=0.25,
                    help='Minimum keep ratio P_0 (default: 0.25)')
    ap.add_argument('--p_step',   type=float, default=0.025,
                    help='Step between P_values (default: 0.025 → 31 subnets)')
    ap.add_argument('--n_importance_samples', type=int, default=1024,
                    help='Number of per-sample backward passes (default: 1024)')
    ap.add_argument('--batch_size', type=int, default=64,
                    help='DataLoader batch size (default: 64)')
    ap.add_argument('--grouping_mode', default='stage',
                    choices=['stage', 'per_block', 'global', 'per_block_perpipe'],
                    help='Formula 5 allocation grouping: stage | per_block | global | per_block_perpipe')
    args = ap.parse_args()

    # Add U-ViT root to sys.path so imports (sde, utils, datasets) work
    uvit_root = os.path.dirname(os.path.abspath(__file__))
    if uvit_root not in sys.path:
        sys.path.insert(0, uvit_root)
    import utils as uvit_utils

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)

    # ── Build model and load weights ─────────────────────────────────────────
    model = uvit_utils.get_nnet(**config.nnet)

    if os.path.isfile(args.ckpt):
        ckpt_file = args.ckpt
    else:
        ckpt_file = os.path.join(args.ckpt, 'nnet_ema.pth')
        if not os.path.isfile(ckpt_file):
            ckpt_file = os.path.join(args.ckpt, 'nnet.pth')
    print(f'Loading weights from {ckpt_file}')

    sd = torch.load(ckpt_file, map_location='cpu', weights_only=True)
    if any(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device)

    # ── Dataset path (CLI overrides config) ──────────────────────────────────
    dataset_path = args.dataset_path or config.dataset.path
    print(f'Dataset path: {dataset_path}')

    # ── Compute Taylor importance ────────────────────────────────────────────
    print(f'\nComputing Taylor importance (Formula 4, VP-SDE LSimple, '
          f'{args.n_importance_samples} samples) …')
    ffn_imp, attn_imp, num_heads, head_dim, hidden_dim = \
        compute_taylor_importance(
            model, config,
            dataset_path         = dataset_path,
            n_importance_samples = args.n_importance_samples,
            batch_size           = args.batch_size,
            device               = device,
        )
    embed_dim = num_heads * head_dim
    print(f'  embed_dim={embed_dim}, num_heads={num_heads}, '
          f'head_dim={head_dim}, hidden_dim={hidden_dim}')

    # ── Build P_values ────────────────────────────────────────────────────────
    P_values = [
        round(v, 8)
        for v in np.arange(args.p_min, 1.0 + args.p_step / 2, args.p_step)
    ]
    # Ensure 1.0 is included
    if P_values[-1] < 1.0:
        P_values.append(1.0)
    print(f'\nP_values ({len(P_values)} subnets, step={args.p_step}): {P_values}')

    # ── Build and save masks ──────────────────────────────────────────────────
    ffn_ranks, attn_ranks, all_masks = build_masks(
        ffn_imp, attn_imp, P_values,
        num_heads, head_dim, hidden_dim,
        p_min=args.p_min,
        grouping_mode=args.grouping_mode)

    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, 'uvit_masks.pt')
    torch.save({
        'masks'             : all_masks,
        'ffn_internal_ranks': ffn_ranks,
        'attn_channel_ranks': attn_ranks,
        'P_values'          : P_values,
        'num_heads'         : num_heads,
        'head_dim'          : head_dim,
        'hidden_dim'        : hidden_dim,
        'embed_dim'         : embed_dim,
        'grouping_mode'     : args.grouping_mode,
    }, save_path)

    print(f'\nSaved {len(P_values)} subnetwork masks → {save_path}')
    print(f'  ffn_internal_ranks: {len(ffn_ranks)} blocks')
    print(f'  attn_channel_ranks: {len(attn_ranks)} blocks  (per-head-offset)')

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f'\n{"P_i":>8}  {"FFN keep":>10}  {"FFN %":>7}  '
          f'{"Attn keep":>10}  {"Attn %":>7}  {"Attn width":>11}')
    print('-' * 65)
    for P in P_values:
        m = all_masks[P]
        ffn_keeps  = [m[b]['ffn_keep']  for b in m if 'ffn_keep'  in m[b]]
        attn_keeps = [m[b]['attn_keep'] for b in m if 'attn_keep' in m[b]]
        avg_ffn   = sum(ffn_keeps) / len(ffn_keeps)   if ffn_keeps  else hidden_dim
        avg_attn  = sum(attn_keeps)/ len(attn_keeps)   if attn_keeps else head_dim
        avg_width = avg_attn * num_heads
        print(f'{P:>8.4f}  {avg_ffn:>10.0f}  {avg_ffn/hidden_dim:>6.1%}  '
              f'{avg_attn:>10.1f}  {avg_attn/head_dim:>6.1%}  {avg_width:>11.0f}')


if __name__ == '__main__':
    main()
