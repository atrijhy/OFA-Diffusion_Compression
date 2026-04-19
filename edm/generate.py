#!/usr/bin/env python3
"""
ofa_generate_physical.py  —  Generate images from OFA physical-slicing models.

Unlike ofa_generate.py (hook-based), this script:
  1. Loads a network-snapshot-*.pkl whose 'ema' is a SliceVPPrecond
  2. Loads physical masks (ofa_masks_physical.pt)
  3. Passes subnet_cfg through network.forward() — NO hooks
  4. Uses EDM's edm_sampler (Heun 2nd-order, Karras schedule)

Usage (single GPU):
  cd EDM/code
  PYTHONPATH=../third_party/edm:$PYTHONPATH \\
  python ofa_generate_physical.py \\
      --network  outputs/ofa_physical/00000-*/network-snapshot-*.pkl \\
      --masks    outputs/ofa_masks_physical/ofa_masks_physical.pt \\
      --p_i      0.5 \\
      --outdir   fid-tmp-physical/p0.5 \\
      --seeds    0-49999 \\
      --batch    64 --steps 18

Multi-GPU:
  torchrun --standalone --nproc_per_node=4 ofa_generate_physical.py \\
      --network ... --masks ... --p_i 0.5 --outdir ... --seeds 0-49999 --subdirs --batch 64
"""

import os
import re
import sys
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

# Import samplers from EDM
from generate import edm_sampler, ablation_sampler, StackedRandomGenerator

# Make SlicePrecond classes visible to pickle
import networks_ofa  # noqa: F401


# ---------------------------------------------------------------------------
# Physical-slicing subnet config helpers (same as ofa_train_edm_physical.py)
# ---------------------------------------------------------------------------

def _load_masks(masks_path):
    raw = torch.load(masks_path, map_location='cpu', weights_only=False)
    all_masks = raw['masks']
    conv_ranks = raw.get('conv_internal_ranks', {})
    qkv_ranks = raw.get('qkv_channel_ranks', {})
    P_values = sorted(all_masks.keys())
    return all_masks, conv_ranks, qkv_ranks, P_values


def _build_subnet_cfg(all_masks, P_i, conv_ranks, qkv_ranks, device):
    blk_masks = all_masks[P_i]
    subnet_cfg = {}
    for blk_name, blk_cfg in blk_masks.items():
        cfg = {}

        if 'conv_keep' in blk_cfg and blk_name in conv_ranks:
            k = blk_cfg['conv_keep']
            cfg['conv_keep_idx'] = conv_ranks[blk_name][:k].sort()[0].to(device)

        if 'qkv_keep' in blk_cfg and blk_name in qkv_ranks:
            kq = blk_cfg['qkv_keep']
            cfg['qkv_keep_idx'] = qkv_ranks[blk_name][:kq].sort()[0].to(device)

        if cfg:
            key = blk_name
            if key.startswith('model.'):
                key = key[len('model.'):]
            subnet_cfg[key] = cfg
    return subnet_cfg


# ---------------------------------------------------------------------------
# Custom sampler wrapper that passes subnet_cfg
# ---------------------------------------------------------------------------

def edm_sampler_physical(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    subnet_cfg=None, **kwargs,
):
    """edm_sampler but passes subnet_cfg through to net(...)."""
    # We wrap net to inject subnet_cfg
    class _NetWithCfg:
        """Thin wrapper: net(x, sigma, labels) → net(x, sigma, labels, subnet_cfg=...)"""
        def __init__(self, net, subnet_cfg):
            self._net = net
            self._cfg = subnet_cfg
            # Copy attributes that edm_sampler reads
            self.sigma_min = net.sigma_min
            self.sigma_max = net.sigma_max
            self.round_sigma = net.round_sigma
            self.img_channels = net.img_channels
            self.img_resolution = net.img_resolution
            self.label_dim = net.label_dim

        def __call__(self, *args, **kw):
            kw['subnet_cfg'] = self._cfg
            return self._net(*args, **kw)

    if subnet_cfg is not None:
        net_wrapped = _NetWithCfg(net, subnet_cfg)
    else:
        net_wrapped = net

    return edm_sampler(net_wrapped, latents, class_labels, randn_like=randn_like, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    for p in s.split(','):
        m = re.match(r'^(\d+)-(\d+)$', p)
        ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1) if m else [int(p)])
    return ranges


@click.command()
@click.option('--network', 'network_pkl', help='OFA network-snapshot-*.pkl', metavar='PATH', type=str, required=True)
@click.option('--outdir', help='Output image directory', metavar='DIR', type=str, required=True)
@click.option('--seeds', help='Random seeds (e.g. 0-49999)', metavar='LIST', type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs', help='Subdirectory per 1000 seeds', is_flag=True)
@click.option('--class', 'class_idx', help='Class label [default: random]', metavar='INT', type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Max batch size', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--steps', 'num_steps', help='Sampling steps', metavar='INT', type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min', help='Min noise level', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max', help='Max noise level', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
@click.option('--rho', help='Time step exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', help='Stochasticity strength', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', help='Stoch. min noise level', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', help='Stoch. max noise level', metavar='FLOAT', type=float, default=float('inf'), show_default=True)
@click.option('--S_noise', help='Stoch. noise inflation', metavar='FLOAT', type=float, default=1, show_default=True)
# ── OFA physical-slicing ─────────────────────────────────────────────────────
@click.option('--masks', 'masks_path', help='Path to ofa_masks_physical.pt', metavar='PT', type=str, required=True)
@click.option('--p_i', help='Subnetwork keep-ratio to sample', metavar='FLOAT', type=float, required=True)
def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size,
         masks_path, p_i, device=torch.device('cuda'), **sampler_kwargs):
    """Generate images from an OFA physical-slicing subnetwork.

    Uses subnet_cfg (physical channel/head slicing) instead of hooks.
    Then use edm/fid.py unchanged for FID computation.
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network
    dist.print0(f'Loading OFA network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    net.eval().requires_grad_(False)

    # Load physical masks
    dist.print0(f'Loading physical masks from "{masks_path}", P_i={p_i}...')
    all_masks, conv_ranks, qkv_ranks, P_values = _load_masks(masks_path)

    P_resolved = min(P_values, key=lambda p: abs(p - p_i))
    if abs(P_resolved - p_i) > 1e-4:
        dist.print0(f'[OFA] Warning: requested P_i={p_i}, using nearest P_i={P_resolved}')
    p_i = P_resolved
    subnet_cfg = _build_subnet_cfg(all_masks, p_i, conv_ranks, qkv_ranks, device)
    dist.print0(f'[OFA] subnet_cfg has {len(subnet_cfg)} blocks (P_i={p_i})')

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Generate
    dist.print0(f'Generating {len(seeds)} images (P_i={p_i}) to "{outdir}"...')
    sampler_kwargs = {k: v for k, v in sampler_kwargs.items() if v is not None}
    # click normalizes option names to lowercase with underscores, while
    # edm_sampler expects S_* keyword names.
    for low_k, up_k in {
        's_churn': 'S_churn',
        's_min': 'S_min',
        's_max': 'S_max',
        's_noise': 'S_noise',
    }.items():
        if low_k in sampler_kwargs:
            sampler_kwargs[up_k] = sampler_kwargs.pop(low_k)

    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels,
                             net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        with torch.no_grad():
            images = edm_sampler_physical(
                net, latents, class_labels, randn_like=rnd.randn_like,
                subnet_cfg=subnet_cfg, **sampler_kwargs,
            )

        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed - seed % 1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    torch.distributed.barrier()
    dist.print0('Done.')


if __name__ == '__main__':
    main()
