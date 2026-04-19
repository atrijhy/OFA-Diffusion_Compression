# prune.py  —  EDM OFA pruning with physical-slicing-aware importance
#
# This script replaces ddpm_prune.py's OFA path.  Key differences:
#
# 1. conv0 + conv1 are treated as ONE "conv internal pipe" — their importance
#    is aggregated per internal channel (same channel index in both layers).
#    Output: `conv_internal_ranks[blk_name]` = sorted indices [out_channels].
#
# 2. qkv is treated as an "attention pipe" — importance is aggregated per
#    within-head OFFSET (sum Q/K/V at same offset across all heads).
#    This ensures all heads keep the same offsets → multi-head safe.
#    Output: `qkv_channel_ranks[blk_name]` = sorted indices [head_dim].
#
# 3. Formula 5 allocates C_l (internal working width) per Layer.  C_l is
#    shared by conv and qkv conceptually, but their ranks arrays have
#    different shapes (conv: [out_ch], qkv: [head_dim = out_ch/num_heads]),
#    so the keep count from each array differs:
#      conv_keep = C_l               (from conv_internal_ranks[:C_l])
#      qkv_keep  = round(C_l / nh)   (from qkv_channel_ranks[:qkv_keep])
#    Total attention width = nh * qkv_keep ≈ C_l  ✓
#
# 4. Mask format:
#    {
#      'masks': {
#        P_i: {
#          'model.enc.16x16_block0': {'conv_keep': 180, 'qkv_keep': 180},
#          'model.dec.32x32_block0': {'conv_keep': 90},  # no attention
#          ...
#        }
#      },
#      'conv_internal_ranks': {blk_name: LongTensor[out_ch] sorted desc},
#      'qkv_channel_ranks':   {blk_name: LongTensor[head_dim] sorted desc},
#    }
#
# Usage:
#   python ddpm_prune_physical.py --pruner ofa \
#       --model_path edm-cifar10-32x32-uncond-vp.pkl \
#       --edm_repo /path/to/edm \
#       --dataset cifar10 --save_path outputs/ofa_masks_physical

import os
import sys
import math
import pickle
import argparse
from collections import defaultdict

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="EDM OFA pruning (physical slicing)")
parser.add_argument("--model_path", type=str, required=True, help="Path to EDM .pkl model")
parser.add_argument("--edm_repo", type=str, required=True, help="Path to NVlabs/edm repo")
parser.add_argument("--dataset", type=str, default=None, help="Dataset path or name")
parser.add_argument("--save_path", type=str, required=True, help="Output directory for masks")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--importance_samples", type=int, default=1024,
                    help="Number of (x0,t) samples for per-sample Taylor importance")
parser.add_argument("--p_min", type=float, default=0.25, help="Min channel keep ratio P_0")
parser.add_argument("--n_subnets", type=int, default=28, help="Number of OFA subnetworks")
parser.add_argument("--pruner", type=str, default="ofa", choices=["ofa"])
parser.add_argument("--precond", type=str, default="vp", choices=["vp", "ve", "edm"],
                    help="Preconditioning type (must match the loaded model)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

sys.path.insert(0, args.edm_repo)
from training.loss import VPLoss, VELoss, EDMLoss
import utils  # Diff-Pruning utils for dataset loading

print(f"[EDM Physical] Loading model from {args.model_path}")
with open(args.model_path, "rb") as f:
    edm_net = pickle.load(f)["ema"].to(args.device).eval()

# Dataset
dataset = utils.get_dataset(args.dataset)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

# Select loss function matching the model's preconditioning
if args.precond == 'vp':
    loss_fn = VPLoss()
elif args.precond == 've':
    loss_fn = VELoss()
else:  # edm
    loss_fn = EDMLoss()
print(f"[EDM Physical] Using {type(loss_fn).__name__} (--precond {args.precond})")
img_ch = 3


# ---------------------------------------------------------------------------
# Step 1: Discover prunable layers and group into blocks + pipes
# ---------------------------------------------------------------------------
# We need conv0, conv1, qkv, proj from each UNetBlock.
# Block name = parent module path (e.g. "model.enc.16x16_block0")
# Two pipes, each measured from BOTH ends (Taylor importance):
#   Conv pipe:  conv0 (output channel i) + conv1 (input channel i)
#   Attn pipe:  qkv  (output offset j)   + proj  (input offset j)

edm_net.requires_grad_(True)

# Find all conv0, conv1, qkv modules with gradients
# First pass: accumulate one batch of gradients to identify which params have grads
edm_net.zero_grad()
it = iter(loader)
batch = next(it)
batch = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(args.device)
loss_fn(net=edm_net, images=batch, labels=None).sum().div(batch.shape[0]).backward()

blocks = {}  # blk_name -> {'conv0': module, 'conv1': module, 'qkv': module (optional)}
named_mods = dict(edm_net.named_modules())

for name, m in edm_net.named_modules():
    if not (hasattr(m, 'weight') and m.weight is not None and
            getattr(m.weight, 'grad', None) is not None):
        continue
    out_ch = getattr(m, 'out_channels', getattr(m, 'out_features', None))
    if out_ch is None or out_ch <= img_ch:
        continue

    leaf = name.split('.')[-1]
    if leaf not in ('conv0', 'conv1', 'qkv', 'proj'):
        continue

    # Exclude up/down resamplers, skip connections, aux
    if any(kw in name for kw in ('_down', '_up', 'skip', 'aux')):
        continue

    if m.weight.dim() not in (2, 4):
        continue

    blk_name = name.rsplit('.', 1)[0]
    if blk_name not in blocks:
        blocks[blk_name] = {}
    blocks[blk_name][leaf] = (name, m)

print(f"[EDM Physical] Found {len(blocks)} blocks:")
for blk, layers in sorted(blocks.items()):
    layers_str = ', '.join(sorted(layers.keys()))
    print(f"  {blk}: {layers_str}")


# ---------------------------------------------------------------------------
# Step 2: Per-sample Taylor importance (Formula 4)
# ---------------------------------------------------------------------------
# For each layer, compute per-OUTPUT-channel importance:
#   I_i^C = (1/N) * Σ_n |Σ_j w_ij * g_ij|
# where the sum Σ_j is over input channels & kernel dims,
# and abs is OUTSIDE the sum (Formula 4).

all_layer_names = []
all_layer_modules = []
for blk, layers in blocks.items():
    for leaf, (name, m) in layers.items():
        all_layer_names.append(name)
        all_layer_modules.append(m)

# Per output channel importance for all layers
channel_imp = {name: torch.zeros(m.out_channels, device=args.device)
               for name, m in zip(all_layer_names, all_layer_modules)}

# Per INPUT channel importance for conv1 and proj layers (accurate per-sample Taylor)
# Same formula as per-output but sum over the output axis instead:
#   I_input[i] = (1/N) Σ_n |Σ_o w[o,i,:,:] * g[o,i,:,:]|
# conv1: the "back end" of the conv pipe  (same internal channel as conv0 output)
# proj:  the "back end" of the attn pipe  (same attention channel as qkv output)
conv1_input_names = []   # full param names for conv1 modules
conv1_input_mods = []    # conv1 modules
conv1_input_imp = {}     # name -> Tensor[internal_ch]

proj_input_names = []    # full param names for proj modules
proj_input_mods = []     # proj modules
proj_input_imp = {}      # name -> Tensor[out_ch]  (proj input dim = out_channels)

for blk_name, layers in blocks.items():
    if 'conv1' in layers:
        name1, m1 = layers['conv1']
        conv1_input_names.append(name1)
        conv1_input_mods.append(m1)
        in_ch = m1.weight.shape[1]  # internal channel count
        conv1_input_imp[name1] = torch.zeros(in_ch, device=args.device)
    if 'proj' in layers:
        name_p, m_p = layers['proj']
        proj_input_names.append(name_p)
        proj_input_mods.append(m_p)
        in_ch_p = m_p.weight.shape[1]  # = out_channels
        proj_input_imp[name_p] = torch.zeros(in_ch_p, device=args.device)

it2 = iter(loader)
for _ in tqdm(range(args.importance_samples), desc="[EDM Physical] per-sample importance"):
    try:
        s = next(it2)
    except StopIteration:
        it2 = iter(loader)
        s = next(it2)
    s = (s[0] if isinstance(s, (list, tuple)) else s).to(args.device)[:1]
    edm_net.zero_grad()
    loss_fn(net=edm_net, images=s, labels=None).mean().backward()

    # Per output channel importance (Formula 4)
    for name, m in zip(all_layer_names, all_layer_modules):
        if m.weight.dim() == 4:
            channel_imp[name] += (m.weight.detach() * m.weight.grad).sum(dim=(1, 2, 3)).abs()
        else:
            channel_imp[name] += (m.weight.detach() * m.weight.grad).sum(dim=1).abs()

    # Per INPUT channel importance for conv1 (accurate per-sample Taylor)
    for name1, m1 in zip(conv1_input_names, conv1_input_mods):
        if m1.weight.grad is not None:
            # weight: [out, internal, k, k] -> sum over (out, k, k) -> [internal]
            conv1_input_imp[name1] += (m1.weight.detach() * m1.weight.grad).sum(dim=(0, 2, 3)).abs()

    # Per INPUT channel importance for proj (accurate per-sample Taylor)
    for name_p, m_p in zip(proj_input_names, proj_input_mods):
        if m_p.weight.grad is not None:
            # weight: [out_ch, out_ch, 1, 1] -> sum over (out, kh, kw) -> [out_ch]
            proj_input_imp[name_p] += (m_p.weight.detach() * m_p.weight.grad).sum(dim=(0, 2, 3)).abs()

for name in channel_imp:
    channel_imp[name] /= args.importance_samples
for name in conv1_input_imp:
    conv1_input_imp[name] /= args.importance_samples
for name in proj_input_imp:
    proj_input_imp[name] /= args.importance_samples


# ---------------------------------------------------------------------------
# Step 3: Aggregate importance per PIPE
# ---------------------------------------------------------------------------
# Conv pipe: conv0 output channel i = conv1 input channel i
#   I_pipe[i] = I_conv0_output[i] + I_conv1_input[i]
#   Ranks shape: [out_channels]
#
# Attention pipe: qkv output offset j = proj input offset j  (same pattern!)
#   I_offset[j] = Σ_h ( I_Q[h*hd+j] + I_K[nh*hd+h*hd+j] + I_V[2*nh*hd+h*hd+j] )
#               + Σ_h I_proj_input[h*hd+j]
#   Ranks shape: [head_dim]  (all heads share same offsets)

conv_internal_imp = {}    # blk_name -> Tensor[out_ch]
conv_internal_ranks = {}  # blk_name -> LongTensor (sorted desc)
qkv_channel_imp = {}      # blk_name -> Tensor[C]  (C = nh*hd, channel-level)
qkv_channel_ranks = {}    # blk_name -> LongTensor (sorted desc)

for blk_name, layers in blocks.items():
    # ── Conv pipe ─────────────────────────────────────────────────────────
    if 'conv0' in layers and 'conv1' in layers:
        name0, m0 = layers['conv0']
        name1, m1 = layers['conv1']
        out_ch = m0.out_channels  # = m1.in_channels (= internal width)

        # conv0: per output channel importance (already computed in Step 2)
        imp0 = channel_imp[name0]  # [out_ch]

        # conv1: per INPUT channel importance (accurately computed in Step 2 loop)
        imp1_input = conv1_input_imp.get(name1, torch.zeros(out_ch, device=args.device))

        # Aggregate: same physical channel viewed from both sides
        pipe_imp = imp0 + imp1_input
        conv_internal_imp[blk_name] = pipe_imp
        conv_internal_ranks[blk_name] = pipe_imp.argsort(descending=True).cpu()
    elif 'conv0' in layers:
        # Block with only conv0 (no conv1 — shouldn't happen in standard UNetBlock)
        name0, m0 = layers['conv0']
        conv_internal_imp[blk_name] = channel_imp[name0]
        conv_internal_ranks[blk_name] = channel_imp[name0].argsort(descending=True).cpu()

    # ── Attention pipe (per-head-offset, uniform across heads) ─────────
    # For each within-head offset j ∈ [0, hd), aggregate importance across
    # all heads and Q/K/V sections.  Every head will keep the same offsets,
    # ensuring valid multi-head attention regardless of num_heads.
    if 'qkv' in layers:
        name_qkv, m_qkv = layers['qkv']
        parent = named_mods.get(blk_name)
        nh = getattr(parent, 'num_heads', 1) or 1
        hd = m_qkv.out_channels // (3 * nh)

        qkv_imp = channel_imp[name_qkv]  # [3*nh*hd]

        # Aggregate per within-head offset j:
        #   I[j] = Σ_h ( I_Q[h*hd+j] + I_K[nh*hd+h*hd+j] + I_V[2*nh*hd+h*hd+j] )
        offset_imp = torch.zeros(hd, device=args.device)
        for h in range(nh):
            q_base = h * hd
            k_base = nh * hd + h * hd
            v_base = 2 * nh * hd + h * hd
            offset_imp += (qkv_imp[q_base:q_base+hd]
                          + qkv_imp[k_base:k_base+hd]
                          + qkv_imp[v_base:v_base+hd])

        # Add proj INPUT channel importance (back end of attention pipe)
        # proj.weight: [out_ch, out_ch, 1, 1], input channels = [nh*hd]
        # Aggregate per within-head offset j across all heads:
        #   proj_offset[j] = Σ_h I_proj_input[h*hd + j]
        if 'proj' in layers:
            name_proj, _ = layers['proj']
            proj_imp_vec = proj_input_imp.get(name_proj)
            if proj_imp_vec is not None:
                for h in range(nh):
                    offset_imp += proj_imp_vec[h*hd:(h+1)*hd]

        qkv_channel_imp[blk_name] = offset_imp   # [hd]
        qkv_channel_ranks[blk_name] = offset_imp.argsort(descending=True).cpu()

print(f"\n[EDM Physical] Importance computed:")
print(f"  Conv pipes: {len(conv_internal_imp)} blocks  (conv0_output + conv1_input)")
print(f"  Attn pipes: {len(qkv_channel_imp)} blocks  (qkv_output + proj_input, per-head-offset)")


# ---------------------------------------------------------------------------
# Step 3b: Per-Layer importance I_l^L = Σ conv_imp + Σ qkv_imp
# ---------------------------------------------------------------------------
# Formula 5 uses the total importance of the Layer (not per-pipe) to decide
# how many channels C_l to allocate.  C_l is then shared by conv and qkv.

layer_imp = {}   # blk_name -> float (total importance of that Layer)
layer_size = {}  # blk_name -> int   (|l| reference width axis for Formula 5)

for blk_name in blocks:
    imp = 0.0
    size = 0
    if blk_name in conv_internal_imp:
        imp += conv_internal_imp[blk_name].sum().item()
        size = conv_internal_imp[blk_name].shape[0]  # = out_channels
    if blk_name in qkv_channel_imp:
        imp += qkv_channel_imp[blk_name].sum().item()
        if size == 0:
            size = qkv_channel_imp[blk_name].shape[0]
    layer_imp[blk_name] = imp
    layer_size[blk_name] = size


# ---------------------------------------------------------------------------
# Step 4: Formula 5 — budget allocation (Algorithm 1, paper-exact)
# ---------------------------------------------------------------------------
# Paper terminology → EDM code:
#   Block B  = resolution group: D_i (enc), U_i (dec), or M (bottleneck)
#   Layer l  = single UNetBlock (Conv₁ + Conv₂ + optional Attention)
#   K        = number of Layers in the Block
#   |l|      = layer reference width axis used by Formula 5.
#              In this implementation it is taken from the conv internal axis
#              (equivalent to block base/residual width for these UNet blocks).
#   I_l^L    = total importance of Layer l (conv + qkv combined)
#
# Formula 5 distributes budget across Layers within a Block:
#   C_l = max{ min{ (I_l^L / Σ_l' I_l'^L) * K * P_i * |l|,  |l| },  P_0 * |l| }
#
# C_l is the budget on the |l| reference axis.
# Pipe keeps are obtained by axis mapping:
#   keep_pipe = round((C_l / |l|) * pipe_full_dim)
# For current EDM blocks this reduces to legacy formulas.

# ── Build resolution groups: D_i, M, U_i ────────────────────────────────
from collections import OrderedDict

res_groups = OrderedDict()  # group_key -> [blk_name, ...]

for blk_name in sorted(blocks.keys()):
    # e.g. "model.enc.16x16_block0" → enc_16x16
    #      "model.dec.8x8_in0"      → mid_8x8  (M block)
    #      "model.dec.8x8_block0"   → dec_8x8
    parts = blk_name.split('.')        # ['model', 'enc', '16x16_block0']
    enc_dec = parts[1]                 # 'enc' or 'dec'
    res_str = parts[2].split('_')[0]   # '16x16', '8x8', '32x32'
    suffix = '_'.join(parts[2].split('_')[1:])  # 'block0', 'in0', etc.

    if enc_dec == 'dec' and suffix.startswith('in'):
        group_key = f"mid_{res_str}"   # M block (bottleneck)
    elif enc_dec == 'enc':
        group_key = f"enc_{res_str}"   # D_i
    else:
        group_key = f"dec_{res_str}"   # U_i

    res_groups.setdefault(group_key, []).append(blk_name)

print("\n[EDM Physical] Resolution groups (paper's Block B):")
for gk, layer_names in res_groups.items():
    short = [n.split('.')[-1] for n in layer_names]
    print(f"  {gk}: K={len(layer_names)} layers  {short}")

# ── Apply Formula 5 per resolution group ──────────────────────────────────

P_values = [round(args.p_min + i * 0.025, 4) for i in range(args.n_subnets)]
# Always include P=1.0 (full model) so OFA training keeps the full-width subnet.
# Default n_subnets=28 gives max=0.925 which misses 1.0; this guard fixes it.
if P_values[-1] < 1.0:
    P_values.append(1.0)

all_masks = {}
print("\nComputing EDM OFA masks (Algorithm 1, Formula 5, per-resolution-group):")

for P_i in P_values:
    mask = {}

    if P_i >= 1.0:
        for blk_name, layers in blocks.items():
            blk_cfg = {}
            if blk_name in conv_internal_imp:
                blk_cfg['conv_keep'] = conv_internal_imp[blk_name].shape[0]
            if blk_name in qkv_channel_imp:
                blk_cfg['qkv_keep'] = qkv_channel_imp[blk_name].shape[0]
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
            l_imp = layer_imp[blk_name]

            # Formula 5
            norm_l = l_imp / group_total_imp
            C_l_raw = norm_l * K * P_i * l_size
            c_min = max(1, round(args.p_min * l_size))
            C_l = max(min(round(C_l_raw), l_size), c_min)

            # ── GroupNorm alignment ──────────────────────────────────
            # EDM uses GroupNorm with num_groups=32 typically.
            # F.group_norm requires num_channels % num_groups == 0.
            # Our _slice_group_norm falls back to fewer groups, but
            # aligning C_l to a nice divisor avoids degenerate g=1.
            # We align to min(32, l_size) so the group count stays
            # reasonable.  For l_size=128 this means multiples of 32;
            # for l_size=256, multiples of 32 as well.
            gn_groups = 32
            if C_l < l_size:  # only align when actually pruning
                # Keep the c_min lower bound intact: only use aligned value
                # when it does not violate the minimum retention constraint.
                base_C_l = C_l
                aligned = max(gn_groups, (C_l // gn_groups) * gn_groups)
                aligned = min(aligned, l_size)
                if aligned >= c_min:
                    C_l = aligned
                else:
                    C_l = base_C_l

            blk_cfg = {}
            # B-axis semantics: C_l is budget on the layer reference axis |l|.
            # Map C_l to each pipe's own axis via ratio = C_l / |l|.
            # (For current EDM conv pipe, conv_full == |l|, so this is
            # equivalent to conv_keep = C_l.)
            ratio = C_l / l_size if l_size > 0 else 1.0

            # Conv pipe mapping (legacy equivalent kept in comment).
            if blk_name in conv_internal_imp:
                conv_full = conv_internal_imp[blk_name].shape[0]
                conv_keep = max(1, round(ratio * conv_full))
                conv_keep = min(conv_keep, conv_full)
                blk_cfg['conv_keep'] = conv_keep
                # --- Legacy executable code (kept as commented source) ---
                # blk_cfg['conv_keep'] = C_l

            # QKV pipe mapping to per-head-offset axis.
            # Legacy form round(C_l / nh) is equivalent because hd = |l| / nh.
            if blk_name in qkv_channel_imp:
                hd = qkv_channel_imp[blk_name].shape[0]  # head_dim
                qkv_keep = max(1, round(ratio * hd))
                qkv_keep = min(qkv_keep, hd)
                blk_cfg['qkv_keep'] = qkv_keep
                # --- Legacy executable code (kept as commented source) ---
                # nh = l_size // hd if hd > 0 else 1       # num_heads
                # qkv_keep = max(1, round(C_l / nh))
                # qkv_keep = min(qkv_keep, hd)
                # blk_cfg['qkv_keep'] = qkv_keep

            mask[blk_name] = blk_cfg

    all_masks[P_i] = mask

    # Stats
    ratios = []
    for blk_name in blocks:
        if blk_name not in mask:
            continue
        blk_cfg = mask[blk_name]
        if 'conv_keep' in blk_cfg and blk_name in conv_internal_imp:
            ratios.append(blk_cfg['conv_keep'] / conv_internal_imp[blk_name].shape[0])
        if 'qkv_keep' in blk_cfg and blk_name in qkv_channel_imp:
            ratios.append(blk_cfg['qkv_keep'] / qkv_channel_imp[blk_name].shape[0])
    avg = sum(ratios) / len(ratios) if ratios else 1.0
    mn = min(ratios) if ratios else 1.0
    print(f"  P_i={P_i:.4f}  avg_keep={avg:.3f}  min={mn:.3f}")


# ---------------------------------------------------------------------------
# Step 5: Save
# ---------------------------------------------------------------------------
# Format:
#   {
#     'masks': {P_i: {blk_name: {'conv_keep': int, 'qkv_keep': int}}},
#     'conv_internal_ranks':  {blk_name: LongTensor[out_ch] sorted desc},
#     'qkv_channel_ranks':   {blk_name: LongTensor[head_dim] sorted desc},
#   }
# conv_keep = C_l (from Formula 5).  Use: conv_ranks[:conv_keep]
# qkv_keep  = round(C_l / num_heads). Use: qkv_ranks[:qkv_keep]
# Both derive from the same C_l, but ranks arrays have different lengths.

os.makedirs(args.save_path, exist_ok=True)
save_path = os.path.join(args.save_path, "ofa_masks_physical.pt")
torch.save({
    'masks': all_masks,
    'conv_internal_ranks': conv_internal_ranks,
    'qkv_channel_ranks': qkv_channel_ranks,
}, save_path)

print(f"\nSaved {len(P_values)} subnetwork masks → {save_path}")
print(f"  conv_internal_ranks:  {len(conv_internal_ranks)} blocks")
print(f"  qkv_channel_ranks:   {len(qkv_channel_ranks)} blocks (channel-level)")
print(f"\nUsage with ofa_train_edm_physical.py:")
print(f"  torchrun --standalone --nproc_per_node=4 ofa_train_edm_physical.py \\")
print(f"      --outdir outputs/ofa_physical \\")
print(f"      --data datasets/cifar10-32x32.zip \\")
print(f"      --masks {save_path} \\")
print(f"      --transfer {args.model_path} \\")
print(f"      --precond vp --arch ddpmpp \\")
print(f"      --batch 512 --lr 1e-3 --duration 102.4")
