#!/usr/bin/env python3
"""Final FID evaluation for a physically extracted (pruned) U-ViT subnet.

Used after uvit_finetune_subnet.py to run official-standard evaluation on the
finetuned pruned model.  The extracted subnet has smaller weight tensors than
the full model; this script uses reshape_to_pruned() to rebuild the correct
architecture before loading the checkpoint.

Default settings match the official U-ViT CIFAR-10 evaluation:
  sample_steps=1000, algorithm=euler_maruyama_sde, n_samples=50000

Usage (single GPU):
  python uvit_eval_extracted_subnet.py \\
      --config    configs/cifar10_uvit_small.py \\
      --extracted outputs/extracted/subnet_0p2500/model.pth \\
      --ckpt      outputs/finetuned/p0p2500/ckpts/200000.ckpt \\
      --outdir    outputs/eval_final

Usage (multi-GPU via accelerate):
  accelerate launch --multi_gpu --num_processes 4 \\
      uvit_eval_extracted_subnet.py ...

Output:
  Prints  "fid=<value>"  on stdout (grep-friendly for the shell pipeline).
  Appends result to  <outdir>/fid_results.txt
"""

import os
import sys

import torch
import accelerate
from absl import app, flags, logging
from ml_collections import config_flags
import ml_collections

import sde
import utils
from datasets import get_dataset
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from tools.fid_score import calculate_fid_given_paths
from extract import reshape_to_pruned


def evaluate(config, extracted_path: str, ckpt_path: str,
             n_samples: int, sample_steps: int, algorithm: str, outdir: str):

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)

    config = ml_collections.FrozenConfigDict(config)

    dataset = get_dataset(**config.dataset)

    # ── Load extracted subnet metadata ────────────────────────────────────────
    ckpt_meta = torch.load(extracted_path, map_location='cpu', weights_only=False)
    P_i            = ckpt_meta['P_i']
    per_block_dims = ckpt_meta['per_block_dims']
    n_params       = ckpt_meta.get('n_params', 0)
    n_params_full  = ckpt_meta.get('n_params_full', 0)

    logging.info(f'P_i={P_i:.4f},  {n_params/1e6:.2f}M / {n_params_full/1e6:.2f}M params')

    # ── Build full model → reshape to pruned architecture → load weights ──────
    config_nnet = ckpt_meta.get('config_nnet', config.nnet)
    nnet_ema = utils.get_nnet(**config_nnet)
    reshape_to_pruned(nnet_ema, per_block_dims)

    if ckpt_path is None:
        # No finetune ckpt → use extracted model weights directly
        sd = ckpt_meta['state_dict']
        if any(k.startswith('module.') for k in sd):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        nnet_ema.load_state_dict(sd, strict=True)
        logging.info(f'No --ckpt provided; using extracted model weights directly')
    else:
        ema_pth = os.path.join(ckpt_path, 'nnet_ema.pth')
        assert os.path.isfile(ema_pth), f'nnet_ema.pth not found in {ckpt_path}'
        sd = torch.load(ema_pth, map_location='cpu', weights_only=True)
        if any(k.startswith('module.') for k in sd):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        nnet_ema.load_state_dict(sd, strict=True)
        logging.info(f'Loaded finetuned EMA weights from {ema_pth}')

    nnet_ema = accelerator.prepare(nnet_ema)
    nnet_ema.eval()

    score_model_ema = sde.ScoreModel(
        accelerator.unwrap_model(nnet_ema), pred=config.pred, sde=sde.VPSDE())

    # ── Sampling ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_fn(_n_samples):
        _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
        kwargs  = {} if config.train.mode == 'uncond' else \
                  dict(y=dataset.sample_label(_n_samples, device=device))
        # No subnet_cfg — the pruned model IS the subnet
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

    p_str   = f'{P_i:.4f}'.replace('.', 'p')
    img_dir = os.path.join(outdir, f'samples_{p_str}')
    if accelerator.is_main_process:
        os.makedirs(img_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logging.info(f'Generating {n_samples} images '
                 f'(algorithm={algorithm}, steps={sample_steps}) …')
    utils.sample2dir(accelerator, img_dir, n_samples,
                     config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)

    # ── FID ───────────────────────────────────────────────────────────────────
    fid = 0.0
    if accelerator.is_main_process:
        fid_stats_available = os.path.exists(dataset.fid_stat)
        if fid_stats_available:
            fid = calculate_fid_given_paths((dataset.fid_stat, img_dir))
            logging.info(f'P_i={P_i:.4f}  fid={fid:.4f}')
            print(f'fid={fid:.4f}', flush=True)

            result_file = os.path.join(outdir, 'fid_results.txt')
            with open(result_file, 'a') as f:
                print(f'P_i={P_i:.4f}  fid={fid:.4f}', file=f)
        else:
            logging.warning(f'FID stats not found: {dataset.fid_stat} — skipping FID')

    return fid


# ── CLI ───────────────────────────────────────────────────────────────────────

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("extracted", None,
                    "Path to extracted subnet model.pth (from extract.py).")
flags.DEFINE_string("ckpt", None,
                    "Finetuned checkpoint directory (e.g. ckpts/200000.ckpt).")
flags.DEFINE_integer("n_samples",    50000, "Number of images to generate.")
flags.DEFINE_integer("sample_steps", 1000,  "Sampling steps (official: 1000).")
flags.DEFINE_string("algorithm", "euler_maruyama_sde",
                    "Sampling algorithm: euler_maruyama_sde | euler_maruyama_ode | dpm_solver.")
flags.DEFINE_string("outdir",  None, "Output directory for images and FID results.")
flags.DEFINE_string("dataset_path", None, "Override dataset path from config.")
flags.mark_flags_as_required(["extracted"])


def main(argv):
    config = FLAGS.config
    if FLAGS.dataset_path:
        config.dataset.path = FLAGS.dataset_path

    from pathlib import Path
    config_name = 'unknown'
    for arg in sys.argv[1:]:
        if arg.startswith('--config='):
            config_name = Path(arg.split('=', 1)[-1]).stem
            break

    outdir = FLAGS.outdir or os.path.join('outputs', f'{config_name}_eval_final')
    os.makedirs(outdir, exist_ok=True)
    utils.set_logger(log_level='info', fname=os.path.join(outdir, 'eval.log'))

    evaluate(
        config         = config,
        extracted_path = FLAGS.extracted,
        ckpt_path      = FLAGS.ckpt,
        n_samples      = FLAGS.n_samples,
        sample_steps   = FLAGS.sample_steps,
        algorithm      = FLAGS.algorithm,
        outdir         = outdir,
    )


if __name__ == '__main__':
    app.run(main)
