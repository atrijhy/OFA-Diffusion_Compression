# train_ofa.py  —  OFA training with physical slicing (no hooks)
#
# Drop-in replacement for ofa_train_edm.py.  Key differences:
#   1. Uses networks_ofa.py (SliceVPPrecond/SliceSongUNet/SliceUNetBlock)
#   2. Loads physical-slicing masks from ddpm_prune_physical.py
#   3. Passes subnet_cfg dict to network forward() — NO forward hooks
#   4. All EDM infrastructure (dist, EMA, checkpointing, stats) is untouched
#
# Usage (4-GPU, CIFAR-10, VP precond):
#   torchrun --standalone --nproc_per_node=4 ofa_train_edm_physical.py \
#       --outdir outputs/ofa_physical \
#       --data datasets/cifar10-32x32.zip \
#       --masks outputs/ofa_masks_physical/ofa_masks_physical.pt \
#       --transfer edm-cifar10-32x32-uncond-vp.pkl \
#       --precond vp --arch ddpmpp \
#       --batch 512 --lr 1e-3 --duration 102.4 \
#       --ema 0.5 --dropout 0.13 --augment 0.12

import os
import sys
import copy
import json
import click
import pickle
import torch
import torch.nn as nn
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from training import training_loop as _edm_loop

# Import physical-slicing network classes so persistence can find them
import networks_ofa  # noqa: F401 — makes SliceVPPrecond etc. visible to pickle

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')


# ---------------------------------------------------------------------------
# OFA helpers
# ---------------------------------------------------------------------------

def _load_masks(masks_path: str):
    """Load ofa_masks_physical.pt (physical slicing format).

    Returns:
        all_masks:            {P_i: {blk_name: {'conv_keep': int, 'qkv_keep': int}}}
        conv_internal_ranks:  {blk_name: LongTensor[out_ch] sorted desc}
        qkv_channel_ranks:    {blk_name: LongTensor[head_dim] sorted desc (per-head offsets)}
        P_values:             sorted list of P_i floats
    """
    raw = torch.load(masks_path, map_location='cpu', weights_only=False)
    all_masks = raw['masks']
    conv_ranks = raw.get('conv_internal_ranks', {})
    qkv_ranks = raw.get('qkv_channel_ranks', {})
    P_values = sorted(all_masks.keys())
    return all_masks, conv_ranks, qkv_ranks, P_values


def _build_subnet_cfg(net, all_masks, P_i, conv_ranks, qkv_ranks, device):
    """Build the subnet_cfg dict that SliceSongUNet.forward() expects.

    Mask format: {blk_name: {'conv_keep': int, 'qkv_keep': int}}
      conv_keep = C_l from Formula 5 (index count for conv_internal_ranks)
      qkv_keep  = round(C_l / num_heads) (index count for qkv_channel_ranks)

    Returns:
        {
            'enc.16x16_block0': {
                'conv_keep_idx': LongTensor on device,
                'qkv_keep_idx':  LongTensor on device,   # only if attention
            },
            ...
        }
    """
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
            # Map block name: ddpm_prune_physical saves e.g. "model.enc.16x16_block0"
            # SliceSongUNet.forward expects e.g. "enc.16x16_block0"
            # Strip the "model." prefix if present
            key = blk_name
            if key.startswith('model.'):
                key = key[len('model.'):]
            subnet_cfg[key] = cfg

    return subnet_cfg


# ---------------------------------------------------------------------------
# OFA training loop with physical slicing
# ---------------------------------------------------------------------------

def ofa_training_loop(
    masks_path,          # path to ofa_masks_physical.pt
    **kwargs,            # forwarded verbatim to EDM infrastructure
):
    """OFA training loop with physical slicing (no hooks)."""

    # ── resolve device ────────────────────────────────────────────────────────
    device = kwargs.get('device', torch.device('cuda'))

    # ── load OFA masks ────────────────────────────────────────────────────────
    dist.print0(f'Loading physical OFA masks from {masks_path} ...')
    all_masks, conv_ranks_cpu, qkv_ranks_cpu, P_values = _load_masks(masks_path)
    P_min, P_max = P_values[0], P_values[-1]
    dist.print0(f'  {len(P_values)} subnetworks, P in [{P_min:.4f}, {P_max:.4f}]')

    # ── subnet sampling weights: w(P_min) = m × w(P_max), linearly descending ─
    # Paper Eq.(6): sample P_i ~ P where w_Pi decreases with P_i (m=3)
    raw_w   = {p: 1.0 + 2.0 * (P_max - p) / (P_max - P_min + 1e-8) for p in P_values}
    total_w = sum(raw_w.values())
    subnet_w = {p: v / total_w for p, v in raw_w.items()}
    # Pre-built weight tensor for torch.multinomial
    _subnet_weights = torch.tensor([subnet_w[p] for p in P_values], dtype=torch.float32)

    # ── Pre-build subnet_cfgs for all P values (they're small, just indices) ──
    # We'll move to device later
    subnet_cfgs_cpu = {}
    for P_i in P_values:
        subnet_cfgs_cpu[P_i] = _build_subnet_cfg(
            None, all_masks, P_i, conv_ranks_cpu, qkv_ranks_cpu, device='cpu')

    # ── replicate the EDM training loop ───────────────────────────────────────
    import time, psutil, numpy as np

    # --- pull kwargs (mirror training_loop.training_loop signature) -----------
    run_dir             = kwargs.get('run_dir', '.')
    dataset_kwargs      = kwargs.get('dataset_kwargs', {})
    data_loader_kwargs  = kwargs.get('data_loader_kwargs', {})
    network_kwargs      = kwargs.get('network_kwargs', {})
    loss_kwargs         = kwargs.get('loss_kwargs', {})
    optimizer_kwargs    = kwargs.get('optimizer_kwargs', {})
    augment_kwargs      = kwargs.get('augment_kwargs', None)
    seed                = kwargs.get('seed', 0)
    batch_size          = kwargs.get('batch_size', 512)
    batch_gpu           = kwargs.get('batch_gpu', None)
    total_kimg          = kwargs.get('total_kimg', 200000)
    ema_halflife_kimg   = kwargs.get('ema_halflife_kimg', 500)
    ema_rampup_ratio    = kwargs.get('ema_rampup_ratio', 0.05)
    lr_rampup_kimg      = kwargs.get('lr_rampup_kimg', 10000)
    loss_scaling        = kwargs.get('loss_scaling', 1)
    kimg_per_tick       = kwargs.get('kimg_per_tick', 50)
    snapshot_ticks      = kwargs.get('snapshot_ticks', 50)
    state_dump_ticks    = kwargs.get('state_dump_ticks', 500)
    resume_pkl          = kwargs.get('resume_pkl', None)
    resume_state_dump   = kwargs.get('resume_state_dump', None)
    resume_kimg         = kwargs.get('resume_kimg', 0)
    cudnn_benchmark     = kwargs.get('cudnn_benchmark', True)

    # --- Initialize (verbatim from training_loop.py) --------------------------
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # --- Dataset (verbatim) ---------------------------------------------------
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(
        dataset=dataset_obj, sampler=dataset_sampler,
        batch_size=batch_gpu, **data_loader_kwargs))

    # --- Network (uses SliceSongUNet via aligned Slice*Precond classes) -------
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution,
                            img_channels=dataset_obj.num_channels,
                            label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels,
                                  net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # --- Optimizer, augpipe, DDP, EMA (verbatim) ------------------------------
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs else None
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device],
                                                     find_unused_parameters=True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # --- Resume (verbatim) ----------------------------------------------------
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'), weights_only=False)
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data

    # ── Move subnet configs to device ─────────────────────────────────────────
    subnet_cfgs = {}
    for P_i, cfg in subnet_cfgs_cpu.items():
        subnet_cfgs[P_i] = {}
        for blk_name, blk_cfg in cfg.items():
            subnet_cfgs[P_i][blk_name] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in blk_cfg.items()
            }

    # --- Train loop -----------------------------------------------------------
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    while True:
        # ── OFA step 1: sample P_i by linearly-descending weights (paper Eq.6) ─
        _pidx_t = torch.multinomial(_subnet_weights, 1).to(device)
        torch.distributed.broadcast(_pidx_t, src=0)
        P_i = P_values[_pidx_t.item()]

        # Get the subnet_cfg for this P_i
        cur_subnet_cfg = subnet_cfgs[P_i]

        # Accumulate gradients
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)

                loss = loss_fn(net=ddp, images=images, labels=labels,
                               augment_pipe=augment_pipe, subnet_cfg=cur_subnet_cfg)
                training_stats.report('Loss/loss', loss)

                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights (verbatim + nan guard)
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA (verbatim)
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Maintenance (verbatim) -----------------------------------------------
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"P_i  {P_i:.4f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        if (not done) and dist.should_stop():
            done = True
            dist.print0('Aborting...')

        # Save network snapshot (verbatim)
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe,
                        dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data

        # Save training state (verbatim)
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) \
                and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()),
                       os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs (verbatim)
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(
                dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    dist.print0('Exiting...')
# ---------------------------------------------------------------------------
# CLI: identical to ofa_train_edm.py, but uses Slice* network classes
# ---------------------------------------------------------------------------

def parse_int_list(s):
    import re
    if isinstance(s, list): return s
    ranges = []
    for p in s.split(','):
        m = re.match(r'^(\d+)-(\d+)$', p)
        ranges.extend(range(int(m.group(1)), int(m.group(2))+1) if m else [int(p)])
    return ranges


@click.command()
@click.option('--outdir',    help='Output directory',                metavar='DIR',   type=str, required=True)
@click.option('--data',      help='Dataset zip or folder',           metavar='ZIP|DIR', type=str, required=True)
@click.option('--cond',      help='Class-conditional',               metavar='BOOL',  type=bool, default=False, show_default=True)
@click.option('--arch',      help='Network architecture',            metavar='ddpmpp|ncsnpp|adm', type=click.Choice(['ddpmpp','ncsnpp','adm']), default='ddpmpp', show_default=True)
@click.option('--precond',   help='Preconditioning & loss',          metavar='vp|ve|edm', type=click.Choice(['vp','ve','edm']), default='vp', show_default=True)
@click.option('--duration',  help='Training duration (Mimg)',        metavar='MIMG',  type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',     help='Total batch size',                metavar='INT',   type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu', help='Batch size per GPU',              metavar='INT',   type=click.IntRange(min=1))
@click.option('--lr',        help='Learning rate',                   metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1e-3, show_default=True)
@click.option('--ema',       help='EMA half-life (Mimg)',            metavar='MIMG',  type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',   help='Dropout probability',             metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',   help='Augment probability',             metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',     help='Dataset x-flips',                 metavar='BOOL',  type=bool, default=False, show_default=True)
@click.option('--fp16',      help='Mixed-precision FP16',            metavar='BOOL',  type=bool, default=False, show_default=True)
@click.option('--ls',        help='Loss scaling',                    metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',     help='cuDNN benchmark',                 metavar='BOOL',  type=bool, default=True, show_default=True)
@click.option('--cache',     help='Cache dataset in CPU',            metavar='BOOL',  type=bool, default=True, show_default=True)
@click.option('--workers',   help='DataLoader workers',              metavar='INT',   type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--desc',      help='Extra tag for output dir name',   metavar='STR',   type=str)
@click.option('--nosubdir',  help='No subdirectory for output',      is_flag=True)
@click.option('--tick',      help='Progress print interval (kimg)',  metavar='KIMG',  type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',      help='Snapshot save interval (ticks)', metavar='TICKS', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',      help='State dump interval (ticks)',     metavar='TICKS', type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',      help='Random seed',                     metavar='INT',   type=int)
@click.option('--transfer',  help='Init from pkl (fine-tuning)',     metavar='PKL',   type=str)
@click.option('--resume',    help='Resume from training-state-*.pt', metavar='PT',    type=str)
@click.option('-n','--dry-run', help='Print options and exit',       is_flag=True)
# ── OFA-specific ─────────────────────────────────────────────────────────────
@click.option('--masks',     help='Path to ofa_masks_physical.pt',   metavar='PT',    type=str, required=True)
def main(**kwargs):
    """OFA joint fine-tuning with PHYSICAL SLICING (no hooks).

    Uses SliceUNetBlock for dynamic channel/head slicing during training.
    All subnet configurations share the same full-width weight tensors;
    each forward pass only computes the kept channels/heads physically.

    \b
    Example (CIFAR-10, VP, 4 GPUs):
    torchrun --standalone --nproc_per_node=4 ofa_train_edm_physical.py \\
        --outdir outputs/ofa_physical \\
        --data datasets/cifar10-32x32.zip \\
        --masks outputs/ofa_masks_physical/ofa_masks_physical.pt \\
        --transfer edm-cifar10-32x32-uncond-vp.pkl \\
        --precond vp --arch ddpmpp \\
        --batch 512 --lr 1e-3 --duration 102.4 \\
        --ema 0.5 --dropout 0.13 --augment 0.12
    """
    import re
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset',
                                        path=opts.data, use_labels=opts.cond,
                                        xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam',
                                          lr=opts.lr, betas=[0.9, 0.999], eps=1e-8)

    # Validate dataset
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution
        c.dataset_kwargs.max_size = len(dataset_obj)
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels in dataset.json')
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture — uses Slice* classes from networks_ofa.py
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SliceSongUNet', embedding_type='positional',
                                encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1],
                                model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SliceSongUNet', embedding_type='fourier',
                                encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1],
                                model_channels=128, channel_mult=[2,2,2])
    else:
        raise click.ClickException('ADM architecture not yet supported for physical slicing. '
                                   'Use --arch ddpmpp or ncsnpp.')

    # Precond + loss — uses Slice* Precond classes
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'networks_ofa.SliceVPPrecond'
        c.loss_kwargs.class_name = 'loss_ofa_aligned.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'networks_ofa.SliceVEPrecond'
        c.loss_kwargs.class_name = 'loss_ofa_aligned.VELoss'
    else:
        c.network_kwargs.class_name = 'networks_ofa.SliceEDMPrecond'
        c.loss_kwargs.class_name = 'loss_ofa_aligned.EDMLoss'

    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer / resume (verbatim)
    if opts.transfer is not None:
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Output directory (verbatim)
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name}-{cond_str}-{opts.arch}-{opts.precond}-ofa-physical-gpus{dist.get_world_size()}-batch{c.batch_size}-{dtype_str}'
    if opts.desc:
        desc += f'-{opts.desc}'

    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_ids = [int(re.match(r'^\d+', x).group()) for x in os.listdir(opts.outdir)
                    if os.path.isdir(os.path.join(opts.outdir, x)) and re.match(r'^\d+', x)] \
                   if os.path.isdir(opts.outdir) else []
        c.run_dir = os.path.join(opts.outdir, f'{max(prev_ids, default=-1)+1:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options
    dist.print0()
    dist.print0('OFA Physical-Slicing Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0(f'  masks_path: {opts.masks}')
    dist.print0()

    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'),
                           file_mode='a', should_flush=True)

    # ── Launch OFA training loop ──────────────────────────────────────────────
    ofa_training_loop(masks_path=opts.masks, **c)


if __name__ == '__main__':
    main()
