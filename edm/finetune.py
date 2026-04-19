#!/usr/bin/env python3
"""Fine-tune an extracted (physically pruned) EDM subnet.

After OFA joint training + subnet extraction (edm_extract_subnet.py), this
script fine-tunes the standalone pruned model with standard EDM loss —
no OFA subnet sampling, no slicing, no subnet_cfg.  The pruned model runs the
"full" forward path with permanently reduced internal dimensions.

This is the final step in the OFA-Diffusion pipeline — each subnet is
independently fine-tuned for a small number of additional steps to recover
quality lost during joint training.

Prerequisites:
  1. Extracted subnet pickle from edm_extract_subnet.py.

Usage:
  cd /wherever/OFA/Diff-Pruning
  PYTHONPATH=/wherever/OFA/edm:$PYTHONPATH python edm_finetune_subnet.py \
      --network   outputs/extracted_subnets_edm/subnet_0p5000/network-snapshot.pkl \
      --data      data/cifar10-32x32.zip \
      --outdir    outputs/finetuned_edm/p0.5000 \
      --precond   vp --arch ddpmpp \
      --batch     512 --lr 5e-4 --duration 20.48 \
      --ema 0.5 --dropout 0.13 --augment 0.12

  Multi-GPU:
  torchrun --standalone --nproc_per_node=4 edm_finetune_subnet.py ...
"""

import os
import sys
import copy
import json
import time
import click
import pickle
import psutil
import numpy as np
import torch
import torch.nn as nn

# Ensure EDM & Diff-Pruning are on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EDM_REPO = os.path.join(SCRIPT_DIR, '..', 'edm')
sys.path.insert(0, os.path.abspath(EDM_REPO))
sys.path.insert(0, SCRIPT_DIR)

import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

# Import network classes for pickle compatibility
import networks_ofa  # noqa: F401
import loss_ofa_aligned       # noqa: F401

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')


# ---------------------------------------------------------------------------
# Standard EDM training loop (no OFA — single subnet, no slicing)
# ---------------------------------------------------------------------------

def finetune_training_loop(
    network_pkl,        # path to extracted subnet .pkl
    **kwargs,
):
    """Standard EDM training loop for an extracted (pruned) subnet.

    This is essentially the same as EDM's training_loop.training_loop()
    but loads the network from an extracted subnet pickle instead of
    constructing from kwargs, and does NOT pass subnet_cfg.
    """
    device = kwargs.get('device', torch.device('cuda'))

    # ── pull kwargs (mirror training_loop signature) ──────────────────────────
    run_dir             = kwargs.get('run_dir', '.')
    dataset_kwargs      = kwargs.get('dataset_kwargs', {})
    data_loader_kwargs  = kwargs.get('data_loader_kwargs', {})
    loss_kwargs         = kwargs.get('loss_kwargs', {})
    optimizer_kwargs    = kwargs.get('optimizer_kwargs', {})
    augment_kwargs      = kwargs.get('augment_kwargs', None)
    seed                = kwargs.get('seed', 0)
    batch_size          = kwargs.get('batch_size', 512)
    batch_gpu           = kwargs.get('batch_gpu', None)
    total_kimg          = kwargs.get('total_kimg', 20480)
    ema_halflife_kimg   = kwargs.get('ema_halflife_kimg', 500)
    ema_rampup_ratio    = kwargs.get('ema_rampup_ratio', None)
    lr_rampup_kimg      = kwargs.get('lr_rampup_kimg', 10000)
    loss_scaling        = kwargs.get('loss_scaling', 1)
    kimg_per_tick       = kwargs.get('kimg_per_tick', 50)
    snapshot_ticks      = kwargs.get('snapshot_ticks', 50)
    state_dump_ticks    = kwargs.get('state_dump_ticks', 500)
    cudnn_benchmark     = kwargs.get('cudnn_benchmark', True)

    # ── Initialize ────────────────────────────────────────────────────────────
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

    # ── Dataset ───────────────────────────────────────────────────────────────
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj, rank=dist.get_rank(),
        num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(
        dataset=dataset_obj, sampler=dataset_sampler,
        batch_size=batch_gpu, **data_loader_kwargs))

    # ── Load extracted subnet ─────────────────────────────────────────────────
    dist.print0(f'Loading extracted subnet from "{network_pkl}" ...')
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    net = data['ema']
    P_i = data.get('P_i', None)
    n_params = sum(p.numel() for p in net.parameters())
    dist.print0(f'  P_i={P_i}, {n_params/1e6:.2f}M params')

    net.train().requires_grad_(True).to(device)

    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels,
                                  net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # ── Optimizer, loss, augpipe, DDP, EMA ────────────────────────────────────
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs)
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) \
        if augment_kwargs else None
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device], find_unused_parameters=True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # ── Training loop (standard EDM — no subnet_cfg) ─────────────────────────
    dist.print0(f'Fine-tuning for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    while True:
        # Accumulate gradients
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)

                # No subnet_cfg — the pruned model IS the subnet
                loss = loss_fn(net=ddp, images=images, labels=labels,
                               augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5,
                                 out=param.grad)
        optimizer.step()

        # Update EMA
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg,
                                   cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Maintenance
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and \
                (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
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

        # Save network snapshot
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            save_data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe,
                             dataset_kwargs=dict(dataset_kwargs),
                             P_i=P_i)
            for key, value in save_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    save_data[key] = value.cpu()
                del value
            if dist.get_rank() == 0:
                pkl_path = os.path.join(
                    run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(save_data, f)
            del save_data

        # Save training state
        if (state_dump_ticks is not None) and \
                (done or cur_tick % state_dump_ticks == 0) and \
                cur_tick != 0 and dist.get_rank() == 0:
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(
                dict(training_stats.default_collector.as_dict(),
                     timestamp=time.time())) + '\n')
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
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',
              help='Extracted subnet .pkl from edm_extract_subnet.py',
              metavar='PKL', type=str, required=True)
@click.option('--outdir', help='Output directory',
              metavar='DIR', type=str, required=True)
@click.option('--data', help='Dataset zip or folder',
              metavar='ZIP|DIR', type=str, required=True)
@click.option('--cond', help='Class-conditional',
              metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--arch', help='Network architecture',
              metavar='ddpmpp|ncsnpp',
              type=click.Choice(['ddpmpp', 'ncsnpp']), default='ddpmpp',
              show_default=True)
@click.option('--precond', help='Preconditioning & loss',
              metavar='vp|ve|edm',
              type=click.Choice(['vp', 've', 'edm']), default='vp',
              show_default=True)
@click.option('--duration', help='Training duration (Mimg)',
              metavar='MIMG', type=click.FloatRange(min=0, min_open=True),
              default=20.48, show_default=True)
@click.option('--batch', help='Total batch size',
              metavar='INT', type=click.IntRange(min=1), default=512,
              show_default=True)
@click.option('--batch-gpu', help='Batch size per GPU',
              metavar='INT', type=click.IntRange(min=1))
@click.option('--lr', help='Learning rate',
              metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=5e-4, show_default=True)
@click.option('--ema', help='EMA half-life (Mimg)',
              metavar='MIMG', type=click.FloatRange(min=0), default=0.5,
              show_default=True)
@click.option('--dropout', help='Dropout probability',
              metavar='FLOAT', type=click.FloatRange(min=0, max=1),
              default=0.13, show_default=True)
@click.option('--augment', help='Augment probability',
              metavar='FLOAT', type=click.FloatRange(min=0, max=1),
              default=0.12, show_default=True)
@click.option('--xflip', help='Dataset x-flips',
              metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--ls', help='Loss scaling',
              metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=1, show_default=True)
@click.option('--bench', help='cuDNN benchmark',
              metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--cache', help='Cache dataset in CPU',
              metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--workers', help='DataLoader workers',
              metavar='INT', type=click.IntRange(min=1), default=1,
              show_default=True)
@click.option('--desc', help='Extra tag for output dir name',
              metavar='STR', type=str)
@click.option('--nosubdir', help='No subdirectory for output', is_flag=True)
@click.option('--tick', help='Progress print interval (kimg)',
              metavar='KIMG', type=click.IntRange(min=1), default=50,
              show_default=True)
@click.option('--snap', help='Snapshot save interval (ticks)',
              metavar='TICKS', type=click.IntRange(min=1), default=50,
              show_default=True)
@click.option('--dump', help='State dump interval (ticks)',
              metavar='TICKS', type=click.IntRange(min=1), default=500,
              show_default=True)
@click.option('--seed', help='Random seed', metavar='INT', type=int)
@click.option('-n', '--dry-run', help='Print options and exit', is_flag=True)
def main(**kwargs):
    """Fine-tune an extracted (pruned) EDM subnet.

    The model is loaded from the extracted .pkl and trained with standard EDM
    loss (no OFA, no subnet_cfg, no slicing).  The pruned model runs its
    full forward path with permanently reduced channel dimensions.

    \b
    Example (CIFAR-10, VP, 4 GPUs, 40k steps ≈ 20.48 Mimg):
    torchrun --standalone --nproc_per_node=4 edm_finetune_subnet.py \\
        --network outputs/extracted_subnets_edm/subnet_0p5000/network-snapshot.pkl \\
        --data datasets/cifar10-32x32.zip \\
        --outdir outputs/finetuned_edm/p0.5000 \\
        --precond vp --arch ddpmpp \\
        --batch 512 --lr 5e-4 --duration 20.48
    """
    import re
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset',
        path=opts.data, use_labels=opts.cond,
        xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam',
        lr=opts.lr, betas=[0.9, 0.999], eps=1e-8)

    # Validate dataset
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution
        c.dataset_kwargs.max_size = len(dataset_obj)
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels')
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Loss (same classes, but no subnet_cfg will be passed)
    if opts.precond == 'vp':
        c.loss_kwargs.class_name = 'loss_ofa_aligned.VPLoss'
    elif opts.precond == 've':
        c.loss_kwargs.class_name = 'loss_ofa_aligned.VELoss'
    else:
        c.loss_kwargs.class_name = 'loss_ofa_aligned.EDMLoss'

    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(
            class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(
            xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1,
            translate_frac=1)

    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.ema_rampup_ratio = None  # No ramp-up for fine-tuning
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap,
             state_dump_ticks=opts.dump)

    # Random seed
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Output directory
    P_i_str = ''
    try:
        with dnnlib.util.open_url(opts.network_pkl) as f:
            meta = pickle.load(f)
        P_i_str = f'-p{meta.get("P_i", "?"):.4f}'
    except Exception:
        pass

    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    desc = f'{dataset_name}-{cond_str}-finetune{P_i_str}-gpus{dist.get_world_size()}-batch{c.batch_size}'
    if opts.desc:
        desc += f'-{opts.desc}'

    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_ids = [int(re.match(r'^\d+', x).group())
                    for x in os.listdir(opts.outdir)
                    if os.path.isdir(os.path.join(opts.outdir, x))
                    and re.match(r'^\d+', x)] \
                   if os.path.isdir(opts.outdir) else []
        c.run_dir = os.path.join(opts.outdir,
                                 f'{max(prev_ids, default=-1)+1:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options
    dist.print0()
    dist.print0('Fine-tuning options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0(f'  network_pkl: {opts.network_pkl}')
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

    # ── Launch fine-tuning ────────────────────────────────────────────────────
    finetune_training_loop(network_pkl=opts.network_pkl, **c)


if __name__ == '__main__':
    main()
