#!/usr/bin/env python3
"""OFA joint fine-tuning for U-ViT — dynamic physical slicing edition.

Implements Algorithm 1 from the paper exactly: for each training step, a subnet
P_i is sampled, its keep-indices are looked up from the pre-computed masks, and
the model is forwarded with *physical slicing* (F.linear on sliced weight
rows/columns) rather than mask+hook.

Key differences vs uvit_train_ofa.py (mask+hook):
  - No forward hooks are registered or removed per step.
  - subnet_cfg is passed directly to UViT.forward() via **kwargs through
    sde.ScoreModel → sde.LSimple.
  - Gradients flow ONLY to the kept neurons / heads; unkept weights receive
    zero gradient — exactly matching Algorithm 1's subnetwork construction.
  - Physical slicing reduces the actual FLOP count of each forward pass.

Prerequisites:
  1. Modified libs/uvit.py (SliceAttention, SliceMlp, subnet_cfg-aware UViT).
  2. uvit_masks.pt from uvit_prune.py (same format as before — no changes needed).

Workflow:
  1. python uvit_prune.py --config <cfg> --ckpt <pretrained> --outdir outputs/uvit_masks
  2. accelerate launch uvit_train_ofa_physical.py \\
         --config  configs/cifar10_uvit_small.py \\
         --masks   outputs/uvit_masks/uvit_masks.pt \\
         --transfer <pretrained_ckpt_dir> \\
         --workdir workdir/cifar10_uvit_ofa_physical

Single-GPU:
  python uvit_train_ofa_physical.py --config ... --masks ... --workdir ...
"""

import os
import sys
import builtins
import tempfile

import torch
import torch.nn as nn
import einops
import accelerate
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torch.utils._pytree import tree_map
from tqdm.auto import tqdm
from absl import logging, flags, app
from ml_collections import config_flags
import ml_collections
import wandb

import sde
import utils
from datasets import get_dataset
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from tools.fid_score import calculate_fid_given_paths


# ===========================================================================
# Subnet config builder  (Algorithm 1 — precomputed physical slice indices)
# ===========================================================================

def build_subnet_cfgs(masks: dict, device) -> dict:
    """Precompute per-P_i physical slice configs from uvit_masks.pt.

    For each P_i and each transformer block, computes:
        ffn_keep_idx   : LongTensor [k_ffn]   — fc1 row / fc2 col indices
        attn_keep_idx  : LongTensor [k_attn]  — within-head offsets to keep

    These are passed directly to UViT.forward(subnet_cfg=...) which performs
    F.linear with sliced weight tensors — no hooks, no zeros.

    SliceAttention internally expands attn_keep_idx to qkv_rows and proj_cols
    via _per_head_qkv_rows / _per_head_proj_cols.

    Mask format (from uvit_prune.py):
        masks['masks']              : {P_i: {blk: {'ffn_keep': int, 'attn_keep': int}}}
        masks['ffn_internal_ranks'] : {blk: LongTensor[hidden_dim] sorted desc}
        masks['attn_channel_ranks'] : {blk: LongTensor[head_dim]  sorted desc}
    """
    all_masks   = masks['masks']              # {P_i: {blk: {ffn_keep, attn_keep}}}
    ffn_ranks   = masks['ffn_internal_ranks'] # {blk: LongTensor[hidden_dim]}
    attn_ranks  = masks['attn_channel_ranks'] # {blk: LongTensor[head_dim]}

    subnet_cfgs = {}
    for P_i in masks['P_values']:
        cfg_P = {}
        blk_masks = all_masks[P_i]

        for blk_name, blk_keep in blk_masks.items():
            blk_cfg = {}

            # ── FFN neurons ───────────────────────────────────────────────
            if 'ffn_keep' in blk_keep:
                n_keep   = blk_keep['ffn_keep']
                keep_idx = ffn_ranks[blk_name][:n_keep].sort()[0].to(device)
                blk_cfg['ffn_keep_idx'] = keep_idx

            # ── Attention (per-head-offset) ───────────────────────────────
            if 'attn_keep' in blk_keep:
                n_keep   = blk_keep['attn_keep']
                keep_idx = attn_ranks[blk_name][:n_keep].sort()[0].to(device)
                blk_cfg['attn_keep_idx'] = keep_idx

            cfg_P[blk_name] = blk_cfg

        subnet_cfgs[P_i] = cfg_P

    return subnet_cfgs


# ===========================================================================
# Training
# ===========================================================================

def train_ofa(config, masks_path: str, transfer_ckpt=None):

    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False

    from torch import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # already set

    accelerator = accelerate.Accelerator()
    device      = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root,  exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.init(
            dir     = os.path.abspath(config.workdir),
            project = f'uvit_ofa_physical_{config.dataset.name}',
            config  = config.to_dict(),
            name    = config.hparams,
            job_type= 'train',
            mode    = 'offline',
        )
        utils.set_logger(log_level='info',
                         fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = get_dataset(**config.dataset)
    if not os.path.exists(dataset.fid_stat):
        logging.warning(f'FID stats not found at {dataset.fid_stat}; '
                        f'eval_step will fail if triggered')
    train_dataset = dataset.get_split(
        split='train', labeled=config.train.mode == 'cond')
    train_loader = DataLoader(
        train_dataset, batch_size=mini_batch_size,
        shuffle=True, drop_last=True,
        num_workers=8, pin_memory=True, persistent_workers=True)

    # ── Model / optimiser / state ────────────────────────────────────────────
    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema,
        train_state.optimizer, train_loader)
    lr_scheduler = train_state.lr_scheduler

    # Optional: load pretrained weights before OFA fine-tuning
    ofa_ckpts_exist = (
        os.path.isdir(config.ckpt_root)
        and any(f.endswith('.ckpt') for f in os.listdir(config.ckpt_root))
    )
    if transfer_ckpt and not ofa_ckpts_exist:
        logging.info(f'Loading pretrained nnet weights from {transfer_ckpt} …')
        _load_weights_only(accelerator, nnet, nnet_ema, transfer_ckpt)

    # Resume OFA training if checkpoint exists
    train_state.resume(config.ckpt_root)

    score_model     = sde.ScoreModel(nnet,     pred=config.pred, sde=sde.VPSDE())
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE())

    # ── OFA masks → physical slice configs (Algorithm 1) ────────────────────
    logging.info(f'Loading OFA masks from {masks_path}')
    masks    = torch.load(masks_path, map_location='cpu', weights_only=False)
    P_values = masks['P_values']
    P_min, P_max = P_values[0], P_values[-1]
    logging.info(f'  {len(P_values)} subnetworks, P ∈ [{P_min:.4f}, {P_max:.4f}]')

    # Precompute subnet configs (index tensors on device) for all P_i
    logging.info('Precomputing physical slice configs …')
    subnet_cfgs = build_subnet_cfgs(masks, device)
    logging.info('  done.')

    # Subnet sampling weights: w(P_min) = 5 × w(P_max), linearly descending (paper Eq.6)
    raw_w    = {p: 1.0 + 2.0 * (P_max - p) / (P_max - P_min + 1e-8) for p in P_values}
    total_w  = sum(raw_w.values())
    subnet_w = {p: v / total_w for p, v in raw_w.items()}
    _subnet_weights = torch.tensor([subnet_w[p] for p in P_values], dtype=torch.float32)
    logging.info(f'  subnet_w: { {round(p,4): round(v,4) for p,v in subnet_w.items()} }')

    def get_data_generator():
        while True:
            for data in tqdm(train_loader,
                             disable=not accelerator.is_main_process,
                             desc='epoch'):
                yield data

    data_generator = get_data_generator()

    # ── OFA training step (physical slicing — no hooks) ──────────────────────
    def train_step(_batch):
        # Sample P_i with linearly-descending weights (paper Eq.6); broadcast from rank-0
        _pidx = torch.multinomial(_subnet_weights, 1).to(device)
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(_pidx, src=0)
        P_i = P_values[_pidx.item()]

        optimizer.zero_grad()

        # Pass subnet_cfg through sde.LSimple → ScoreModel.predict → nnet.forward
        # No hooks registered or removed — gradients flow ONLY to kept neurons.
        s_cfg = subnet_cfgs[P_i]
        if config.train.mode == 'uncond':
            loss = sde.LSimple(score_model, _batch, pred=config.pred,
                               subnet_cfg=s_cfg)
        elif config.train.mode == 'cond':
            loss = sde.LSimple(score_model, _batch[0], pred=config.pred,
                               y=_batch[1], subnet_cfg=s_cfg)
        else:
            raise NotImplementedError(config.train.mode)

        _metrics = {
            'loss': accelerator.gather(loss.detach()).mean().item(),
            'P_i' : P_i,
        }

        accelerator.backward(loss.mean())

        if 'grad_clip' in config and config.grad_clip > 0:
            accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.grad_clip)

        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1

        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    # ── Evaluation step ───────────────────────────────────────────────────────
    # subnet_cfg=None  → full EMA model (checkpoint selection + full eval)
    # subnet_cfg=<cfg> → pruned EMA model (subnet quality tracking)
# ── Evaluation step ───────────────────────────────────────────────────────
    # subnet_cfg=None  → full EMA model (checkpoint selection + full eval)
    # subnet_cfg=<cfg> → pruned EMA model (subnet quality tracking)
    def eval_step(n_samples, sample_steps, algorithm, subnet_cfg=None, label=None):
        if label is None:
            label = 'full' if subnet_cfg is None else f'P{P_min:.4f}'
        logging.info(f'eval_step[{label}]: n_samples={n_samples}, '
                     f'sample_steps={sample_steps}, algorithm={algorithm}')

        def sample_fn(_n_samples):
            _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
            # Build base kwargs (conditional generation labels, if any)
            kwargs  = {} if config.train.mode == 'uncond' else \
                      dict(y=dataset.sample_label(_n_samples, device=device))
            # Pass subnet_cfg so ScoreModel → nnet_ema.forward slices the subnet
            if subnet_cfg is not None:
                kwargs['subnet_cfg'] = subnet_cfg
            if algorithm == 'euler_maruyama_sde':
                return sde.euler_maruyama(
                    sde.ReverseSDE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'euler_maruyama_ode':
                return sde.euler_maruyama(
                    sde.ODE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'dpm_solver':
                noise_schedule = NoiseScheduleVP(schedule='linear')
                model_fn = model_wrapper(
                    score_model_ema.noise_pred, noise_schedule,
                    time_input_type='0', model_kwargs=kwargs)
                dpm = DPM_Solver(model_fn, noise_schedule)
                return dpm.sample(_x_init, steps=sample_steps,
                                  eps=1e-4, adaptive_step_size=False,
                                  fast_version=True)
            else:
                raise NotImplementedError(algorithm)

        path = config.sample.path or os.path.join(config.sample_dir, f'eval_tmp_{train_state.step}')
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        accelerator.wait_for_everyone()

        utils.sample2dir(accelerator, path, n_samples,
                         config.sample.mini_batch_size,
                         sample_fn, dataset.unpreprocess)

        _fid = 0
        if accelerator.is_main_process:
            _fid = calculate_fid_given_paths((dataset.fid_stat, path))
            log_key = f'fid{n_samples}_{label}'
            logging.info(f'step={train_state.step} {log_key}={_fid}')
            with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                print(f'step={train_state.step} {log_key}={_fid}', file=f)
            wandb.log({log_key: _fid}, step=train_state.step)

        # Cast to float32 before NCCL all_reduce to avoid crashes on mixed-precision setups
        _fid = torch.tensor(float(_fid), dtype=torch.float32, device=device)
        _fid = accelerator.reduce(_fid, reduction='sum')

        if accelerator.is_main_process and not config.sample.path:
            import shutil
            shutil.rmtree(path, ignore_errors=True)
            
        return _fid.item()

    # ── Main loop ────────────────────────────────────────────────────────────
    logging.info(f'Start OFA physical fine-tuning, step={train_state.step}, '
                 f'mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch   = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if (accelerator.is_main_process
                and train_state.step % config.train.log_interval == 0):
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if (accelerator.is_main_process
                and train_state.step % config.train.eval_interval == 0):
            logging.info('Save a grid of images …')
            x_init = torch.randn(100, *dataset.data_shape, device=device)
            if config.train.mode == 'uncond':
                samples = sde.euler_maruyama(
                    sde.ODE(score_model_ema), x_init=x_init, sample_steps=50)
            elif config.train.mode == 'cond':
                y = einops.repeat(
                    torch.arange(10, device=device) % dataset.K,
                    'nrow -> (nrow ncol)', ncol=10)
                samples = sde.euler_maruyama(
                    sde.ODE(score_model_ema), x_init=x_init,
                    sample_steps=50, y=y)
            else:
                raise NotImplementedError
            samples = make_grid(dataset.unpreprocess(samples), 10)
            save_image(samples,
                       os.path.join(config.sample_dir, f'{train_state.step}.png'))
            wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

        if (train_state.step % config.train.save_interval == 0
                or train_state.step == config.train.n_steps):
            logging.info(f'Save and eval checkpoint {train_state.step} …')
            if accelerator.local_process_index == 0:
                train_state.save(
                    os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            # Quick FID for head (P_min) and tail (P_max) at each save_interval
            _eval_n       = int(config.train.get('eval_n_samples',    config.sample.n_samples))
            _eval_steps   = int(config.train.get('eval_sample_steps', config.sample.sample_steps))
            _eval_algo    = config.train.get('eval_algorithm',        config.sample.algorithm)
            for _p, _lbl in [(P_min, f'P{P_min:.4f}'), (P_max, f'P{P_max:.4f}')]:
                eval_step(n_samples=_eval_n, sample_steps=_eval_steps,
                          algorithm=_eval_algo, subnet_cfg=subnet_cfgs[_p], label=_lbl)
            torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

    logging.info(f'Finish OFA physical fine-tuning, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    if step_fid:
        step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
        logging.info(f'step_best: {step_best}')
        train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))

    accelerator.wait_for_everyone()
    # Final eval: full model + smallest subnet (P_min)
    eval_step(n_samples=config.sample.n_samples,
              sample_steps=config.sample.sample_steps,
              algorithm=config.sample.algorithm)
    eval_step(n_samples=config.sample.n_samples,
              sample_steps=config.sample.sample_steps,
              algorithm=config.sample.algorithm,
              subnet_cfg=subnet_cfgs[P_min])


# ===========================================================================
# Helpers
# ===========================================================================

def _load_weights_only(accelerator, nnet, nnet_ema, ckpt: str):
    """Load nnet weights from a checkpoint directory or a single .pth file."""
    def _load_sd(model, sd):
        if any(k.startswith('module.') for k in sd):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        accelerator.unwrap_model(model).load_state_dict(sd, strict=True)

    if os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location='cpu', weights_only=True)
        _load_sd(nnet,     sd)
        _load_sd(nnet_ema, sd)
        logging.info(f'  loaded weights from {ckpt} → nnet + nnet_ema')
        return

    def _load(model, fname):
        path = os.path.join(ckpt, fname)
        if os.path.isfile(path):
            sd = torch.load(path, map_location='cpu', weights_only=True)
            _load_sd(model, sd)
            logging.info(f'  loaded {fname}')
        else:
            logging.warning(f'  {fname} not found, skipping')

    _load(nnet,     'nnet.pth')
    _load(nnet_ema, 'nnet_ema.pth')

    # Fallback: if nnet.pth not found but nnet_ema.pth exists, copy ema → nnet
    if not os.path.isfile(os.path.join(ckpt, 'nnet.pth')) and \
       os.path.isfile(os.path.join(ckpt, 'nnet_ema.pth')):
        logging.info('  nnet.pth not found; copying nnet_ema weights → nnet')
        sd = torch.load(os.path.join(ckpt, 'nnet_ema.pth'),
                        map_location='cpu', weights_only=True)
        _load_sd(nnet, sd)


# ===========================================================================
# CLI
# ===========================================================================

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir",  None, "Work directory.")
flags.DEFINE_string("masks",    None, "Path to uvit_masks.pt (required).")
flags.DEFINE_string("transfer", None,
                    "Pretrained checkpoint directory to initialise fine-tuning from. "
                    "Weights are loaded only if no OFA checkpoint exists yet in workdir.")
flags.mark_flags_as_required(["masks"])


def _get_config_name():
    from pathlib import Path
    for arg in sys.argv[1:]:
        if arg.startswith('--config='):
            return Path(arg.split('=', 1)[-1]).stem
    return 'unknown'


def _get_hparams():
    from pathlib import Path
    lst = []
    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        if arg.startswith('--config.') and not arg.startswith('--config.dataset.path'):
            k, v = arg.split('=', 1)
            k = k.split('.')[-1]
            if k.endswith('path'):
                v = Path(v).stem
            lst.append(f'{k}={v}')
    return '-'.join(lst) or 'default'


def main(argv):
    config = FLAGS.config
    config.config_name = _get_config_name()
    config.hparams     = _get_hparams()
    config.workdir     = FLAGS.workdir or os.path.join(
        'workdir', config.config_name + '_ofa_physical', config.hparams)
    config.ckpt_root   = os.path.join(config.workdir, 'ckpts')
    config.sample_dir  = os.path.join(config.workdir, 'samples')
    train_ofa(config, masks_path=FLAGS.masks, transfer_ckpt=FLAGS.transfer)


if __name__ == '__main__':
    app.run(main)
