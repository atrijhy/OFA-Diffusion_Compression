#!/usr/bin/env python3
"""Fine-tune an extracted (physically pruned) U-ViT subnet.

After OFA joint training + subnet extraction (extract.py), this
script fine-tunes the standalone pruned model with standard LSimple loss —
no OFA subnet sampling, no slicing, no hooks.  The pruned model runs the
"full" forward path with permanently reduced internal dimensions.

This is the final step in the OFA-Diffusion pipeline — each subnet is
independently fine-tuned for a small number of additional steps to recover
quality lost during joint training.

Prerequisites:
  1. Extracted subnet checkpoint from extract.py.

Usage:
  python uvit_finetune_subnet.py \
      --config      configs/cifar10_uvit_small.py \
      --extracted   outputs/extracted_subnets/subnet_0p5000/model.pth \
      --n_steps     20000 \
      --save_steps  10000,20000 \
      --workdir     outputs/finetuned/p0.5000

  Multi-GPU:
  accelerate launch --multi_gpu --num_processes 4 uvit_finetune_subnet.py ...
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

# Reuse architecture reshaper from extraction script
from extract import reshape_to_pruned


# ===========================================================================
# Training
# ===========================================================================

def finetune_subnet(config, extracted_path: str, n_steps: int,
                    save_steps: list, eval_steps: list, workdir: str,
                    eval_n_samples: int, eval_sample_steps: int,
                    eval_algorithm: str, resume_ckpt: str = None):

    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False

    from torch import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    accelerator = accelerate.Accelerator()
    device      = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision

    ckpt_root  = os.path.join(workdir, 'ckpts')
    sample_dir = os.path.join(workdir, 'samples')

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(ckpt_root,  exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # ── Load extracted subnet ────────────────────────────────────────────────
    logging.info(f'Loading extracted subnet from {extracted_path}')
    ckpt = torch.load(extracted_path, map_location='cpu', weights_only=False)

    P_i            = ckpt['P_i']
    per_block_dims = ckpt['per_block_dims']
    n_params_orig  = ckpt.get('n_params_full', 0)
    n_params_sub   = ckpt.get('n_params', 0)

    logging.info(f'  P_i={P_i:.4f}, '
                 f'{n_params_sub/1e6:.2f}M params '
                 f'(full model: {n_params_orig/1e6:.2f}M)')

    # Create standard UViT → reshape to pruned architecture → load weights
    config_nnet = ckpt.get('config_nnet', config.nnet)
    nnet = utils.get_nnet(**config_nnet)
    reshape_to_pruned(nnet, per_block_dims)
    nnet.load_state_dict(ckpt['state_dict'], strict=True)
    logging.info('  Architecture reshaped + state_dict loaded ✓')

    # EMA copy
    import copy
    nnet_ema = copy.deepcopy(nnet)
    nnet_ema.eval()

    # Freeze config
    config = ml_collections.FrozenConfigDict(config)

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = get_dataset(**config.dataset)
    if not os.path.exists(dataset.fid_stat):
        logging.warning(f'FID stats not found at {dataset.fid_stat}; '
                        f'eval will fail if triggered')
    train_dataset = dataset.get_split(
        split='train', labeled=config.train.mode == 'cond')
    train_loader = DataLoader(
        train_dataset, batch_size=mini_batch_size,
        shuffle=True, drop_last=True,
        num_workers=8, pin_memory=True, persistent_workers=True)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = utils.get_optimizer(nnet.parameters(), **config.optimizer)

    # Simple cosine-warmup LR scheduler
    # _resume_start is a mutable container so lr_lambda can see the final
    # start_step value after the resume checkpoint is loaded below.
    from torch.optim.lr_scheduler import LambdaLR
    warmup = config.lr_scheduler.get('warmup_steps', 0)
    _resume_start = [0]
    def lr_lambda(step):
        abs_step = _resume_start[0] + step
        if abs_step < warmup:
            return abs_step / max(warmup, 1)
        return 1.0
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Accelerate prepare ───────────────────────────────────────────────────
    nnet, nnet_ema, optimizer, train_loader = accelerator.prepare(
        nnet, nnet_ema, optimizer, train_loader)

    score_model     = sde.ScoreModel(nnet,     pred=config.pred, sde=sde.VPSDE())
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE())

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_step = 0
    if resume_ckpt:
        nnet_pth     = os.path.join(resume_ckpt, 'nnet.pth')
        nnet_ema_pth = os.path.join(resume_ckpt, 'nnet_ema.pth')
        assert os.path.isfile(nnet_pth),     f'nnet.pth not found in {resume_ckpt}'
        assert os.path.isfile(nnet_ema_pth), f'nnet_ema.pth not found in {resume_ckpt}'
        accelerator.unwrap_model(nnet).load_state_dict(
            torch.load(nnet_pth,     map_location='cpu', weights_only=True), strict=True)
        accelerator.unwrap_model(nnet_ema).load_state_dict(
            torch.load(nnet_ema_pth, map_location='cpu', weights_only=True), strict=True)
        ckpt_name  = os.path.basename(resume_ckpt.rstrip('/'))   # e.g. "10000.ckpt"
        start_step = int(ckpt_name.replace('.ckpt', ''))
        _resume_start[0] = start_step
        logging.info(f'Resumed from {resume_ckpt}, start_step={start_step}')

    # ── W&B ──────────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        wandb.init(
            dir     = os.path.abspath(workdir),
            project = f'uvit_finetune_{config.dataset.name}',
            config  = {'P_i': P_i, 'n_steps': n_steps,
                       'n_params': n_params_sub,
                       **config.to_dict()},
            name    = f'finetune_p{P_i:.4f}',
            job_type= 'finetune',
            mode    = 'offline',
        )
        utils.set_logger(log_level='info',
                         fname=os.path.join(workdir, 'output.log'))
        logging.info(f'Fine-tuning P_i={P_i:.4f} for {n_steps} steps '
                     f'(saving at {save_steps}, eval at {eval_steps})')
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    # ── Data generator ───────────────────────────────────────────────────────
    def get_data_generator():
        while True:
            for data in tqdm(train_loader,
                             disable=not accelerator.is_main_process,
                             desc='epoch'):
                yield data

    data_generator = get_data_generator()

    # ── EMA helper ───────────────────────────────────────────────────────────
    ema_rate = config.get('ema_rate', 0.9999)

    @torch.no_grad()
    def ema_update():
        for p, p_ema in zip(
                accelerator.unwrap_model(nnet).parameters(),
                accelerator.unwrap_model(nnet_ema).parameters()):
            p_ema.data.mul_(ema_rate).add_(p.data, alpha=1 - ema_rate)

    # ── Standard training step (no OFA — full pruned model) ──────────────────
    def train_step(_batch):
        optimizer.zero_grad()

        # No subnet_cfg — the pruned model IS the subnet
        if config.train.mode == 'uncond':
            loss = sde.LSimple(score_model, _batch, pred=config.pred)
        elif config.train.mode == 'cond':
            loss = sde.LSimple(score_model, _batch[0], pred=config.pred,
                               y=_batch[1])
        else:
            raise NotImplementedError(config.train.mode)

        _metrics = {'loss': accelerator.gather(loss.detach()).mean().item()}

        accelerator.backward(loss.mean())

        if 'grad_clip' in config and config.grad_clip > 0:
            accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.grad_clip)

        optimizer.step()
        lr_scheduler.step()
        ema_update()

        return dict(lr=optimizer.param_groups[0]['lr'], **_metrics)

    # ── Evaluation (FID on EMA model) ────────────────────────────────────────
    def eval_step(step_num, n_samples, sample_steps, algorithm):
        logging.info(f'eval_step @{step_num}: n_samples={n_samples}, '
                     f'sample_steps={sample_steps}, algorithm={algorithm}')

        def sample_fn(_n_samples):
            _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
            kwargs  = {} if config.train.mode == 'uncond' else \
                      dict(y=dataset.sample_label(_n_samples, device=device))
            # No subnet_cfg — the pruned model runs full forward
            if algorithm == 'euler_maruyama_sde':
                return sde.euler_maruyama(
                    sde.ReverseSDE(score_model_ema), _x_init, sample_steps,
                    **kwargs)
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

        path = os.path.join(workdir, f'eval_tmp_{step_num}')
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        accelerator.wait_for_everyone()

        utils.sample2dir(accelerator, path, n_samples,
                         config.sample.mini_batch_size,
                         sample_fn, dataset.unpreprocess)

        _fid = 0
        if accelerator.is_main_process:
            _fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'step={step_num} fid{n_samples}={_fid}')
            print(f'step={step_num} fid={_fid:.4f}', flush=True)
            with open(os.path.join(workdir, 'eval.log'), 'a') as f:
                print(f'step={step_num} fid{n_samples}={_fid}', file=f)
            wandb.log({f'fid{n_samples}': _fid}, step=step_num)
            import shutil
            shutil.rmtree(path, ignore_errors=True)
        _fid = torch.tensor(float(_fid), dtype=torch.float32, device=device)
        _fid = accelerator.reduce(_fid, reduction='sum')
        return _fid.item()

    # ── Save checkpoint ──────────────────────────────────────────────────────
    def save_ckpt(step_num):
        ckpt_dir = os.path.join(ckpt_root, f'{step_num}.ckpt')
        if accelerator.local_process_index == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(accelerator.unwrap_model(nnet).state_dict(),
                       os.path.join(ckpt_dir, 'nnet.pth'))
            torch.save(accelerator.unwrap_model(nnet_ema).state_dict(),
                       os.path.join(ckpt_dir, 'nnet_ema.pth'))
            logging.info(f'Saved checkpoint → {ckpt_dir}')
        accelerator.wait_for_everyone()

    # ── Main loop ────────────────────────────────────────────────────────────
    logging.info(f'Start fine-tuning, mixed_precision={config.mixed_precision}')

    save_steps_set = set(start_step + s for s in save_steps)
    eval_steps_set = set(start_step + s for s in eval_steps)
    step_fid = []
    log_interval  = config.train.get('log_interval', 100)
    eval_interval = config.train.get('eval_interval', 5000)

    for step in range(start_step + 1, start_step + n_steps + 1):
        nnet.train()
        batch   = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and step % log_interval == 0:
            logging.info(utils.dct2str(dict(step=step, **metrics)))
            wandb.log(metrics, step=step)

        if (accelerator.is_main_process
                and step % eval_interval == 0):
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
            save_image(samples, os.path.join(sample_dir, f'{step}.png'))
            wandb.log({'samples': wandb.Image(samples)}, step=step)
            torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

        # Save at configured save steps; eval can be a subset via eval_steps.
        if step in save_steps_set:
            save_ckpt(step)
        if step in eval_steps_set:
            fid = eval_step(step,
                            n_samples=eval_n_samples,
                            sample_steps=eval_sample_steps,
                            algorithm=eval_algorithm)
            step_fid.append((step, fid))
            torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

    # ── Final summary ────────────────────────────────────────────────────────
    logging.info(f'Finish fine-tuning P_i={P_i:.4f}, {n_steps} steps')
    if step_fid:
        logging.info(f'step_fid: {step_fid}')
        best_step, best_fid = min(step_fid, key=lambda x: x[1])
        logging.info(f'Best: step={best_step}, FID={best_fid:.4f}')

        if accelerator.is_main_process:
            with open(os.path.join(workdir, 'fid_summary.txt'), 'w') as f:
                f.write(f'# P_i={P_i:.4f}  n_params={n_params_sub}\n')
                f.write('# step  fid\n')
                for s, fid in step_fid:
                    f.write(f'{s}  {fid:.4f}\n')
                f.write(f'# best: step={best_step}  fid={best_fid:.4f}\n')


# ===========================================================================
# CLI
# ===========================================================================

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("extracted", None,
                    "Path to extracted subnet .pth from extract.py.")
flags.DEFINE_integer("n_steps", 20000, "Total fine-tuning steps.")
flags.DEFINE_string("save_steps", "10000,20000",
                    "Comma-separated steps at which to save + eval FID.")
flags.DEFINE_string("eval_steps", "",
                    "Comma-separated steps at which to evaluate FID. "
                    "If empty, defaults to --save_steps.")
flags.DEFINE_integer("eval_n_samples", 10000,
                    "Number of samples for each save-step FID evaluation.")
flags.DEFINE_integer("eval_sample_steps", 50,
                    "Sampling steps for each save-step FID evaluation.")
flags.DEFINE_string("eval_algorithm", "dpm_solver",
                    "Sampling algorithm for save-step FID evaluation.")
flags.DEFINE_string("workdir", None, "Work directory for checkpoints + logs.")
flags.DEFINE_string("dataset_path", None, "Override dataset path from config.")
flags.DEFINE_string("resume_ckpt", None,
                    "Checkpoint directory to resume from (e.g. ckpts/10000.ckpt). "
                    "Weights are loaded from nnet.pth + nnet_ema.pth; step counter "
                    "starts after the inferred step (parsed from directory name). "
                    "--save_steps are relative to the resume step.")
flags.mark_flags_as_required(["extracted"])


def main(argv):
    config = FLAGS.config

    if FLAGS.dataset_path:
        config.dataset.path = FLAGS.dataset_path

    save_steps = [int(s) for s in FLAGS.save_steps.split(',')]
    eval_steps = ([int(s) for s in FLAGS.eval_steps.split(',')]
                  if FLAGS.eval_steps else list(save_steps))
    n_steps    = FLAGS.n_steps

    ckpt = torch.load(FLAGS.extracted, map_location='cpu', weights_only=False)
    P_i  = ckpt['P_i']

    workdir = FLAGS.workdir or os.path.join(
        'outputs', f'finetune_p{P_i:.4f}')

    finetune_subnet(
        config            = config,
        extracted_path    = FLAGS.extracted,
        n_steps           = n_steps,
        save_steps        = save_steps,
        eval_steps        = eval_steps,
        workdir           = workdir,
        eval_n_samples    = FLAGS.eval_n_samples,
        eval_sample_steps = FLAGS.eval_sample_steps,
        eval_algorithm    = FLAGS.eval_algorithm,
        resume_ckpt       = FLAGS.resume_ckpt,
    )


if __name__ == '__main__':
    app.run(main)
