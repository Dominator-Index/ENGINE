import argparse
import logging
import math
import os
from datetime import datetime

import numpy as np
import torch
import transformers

from common.logging import get_logger
from dataset import get_data_module
from model import get_model
from trainer import finetune_ZO, finetune_FO, finetune_EVVMS


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Model Fine-tuning')

    # common
    parser.add_argument('--model_name', type=str, default='distilbert-base-cased',
                        help='Name of the pre-trained model')
    parser.add_argument('--task', type=str, default='mnli', help='Task for model training')
    parser.add_argument('--algorithm', type=str, default='MeZO', help='Algorithm to use',
                        choices=["FOSGD", "FOAdam", "MeZO", "ZOAdam", "ZOSVRG", "LOZO", "HiZOO", "LOREN", "AdaSmooth", "ENGINE"])
    parser.add_argument('--device', type=int, default=0, help='GPU Number')
    parser.add_argument('--results', type=str, default='results_demo',
                        help='Name of folder to store results')
    # training
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training')
    parser.add_argument('--batchsize_limit', type=int, default=64,
                        help='Max batch size to be used to avoid memory error')
    parser.add_argument('--full_parameter', action='store_true',
                        help='True for full parameter fine-tuning')
    # dataset
    parser.add_argument('--samplesize', type=int, default=512, help='Training data sample size')
    parser.add_argument('--samplesize_validation', type=int, default=256,
                        help='Validation data sample size')
    parser.add_argument('--max_seq_length', type=int, default=256,
                        help='Max sequence length for inputs')
    # optimization
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay')
    parser.add_argument('--perturbation_scale', type=float, default=1e-3,
                        help='Perturbation scale for SPSA estimators')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience; '
                        'if the validation accuracy does not improve for this number of evaluations, '
                        'stop training')
    parser.add_argument('--coarse_patience', type=int, default=0,
                        help='V-cycle coarse grid early stopping patience (0=disabled). '
                        'If val_acc does not improve for this many coarse epochs, advance to fine grid.')
    parser.add_argument('--fine_patience', type=int, default=0,
                        help='V-cycle fine grid early stopping patience (0=disabled). '
                        'If val_acc does not improve for this many fine epochs, advance to next V-cycle.')
    parser.add_argument('--num_queries', type=int, default=10,
                        help='Number of queries to evaluate in large batch update')
    # experiment
    parser.add_argument('--soft_prompt', action='store_true', help='True for using soft prompt')
    parser.add_argument('--save_ckpt', action='store_true', help='Save checkpoint')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every n epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate on validation set every n epochs')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    parser.add_argument('--bf16', action='store_true', help='Use BF16 precision')
    parser.add_argument('--logging', type=str, default="csv", choices=['tensorboard', 'csv', 'wandb', 'wandb+csv'],
                        help="Choose logging method; either wandb or tensorboard or csv")
    parser.add_argument('--trial', type=int, default=0, help='Trial number')
    parser.add_argument('--init_seed', type=int, default=None, help='Random seed for model initialization')
    parser.add_argument('--clipping', action='store_true', help='set to True to apply gradient clipping')
    # SVRG
    parser.add_argument('--q', type=int, default=2, help='q parameter used only for ZO-SVRG')
    parser.add_argument('--lr_mezosvrg_mb', type=float, default=1e-6,
                        help='Mini-batch learning rate for MeZO-SVRG')
    parser.add_argument('--anneal', type=float, default=1.5, help='Annealing parameter')
    # Adam
    parser.add_argument('--eps', type=float, default=1e-8, help='the level of adaptivity (eps for adam)')
    parser.add_argument('--low_bit_adam', type=int, default=0,
                        help='Use Adam with quantized states; options: 4 or 8')
    # LOZO
    parser.add_argument('--rank', type=int, default=2, help='rank of U and V')
    parser.add_argument('--step_interval', type=int, default=50, help='frequency of updating matrix V')
    # HiZOO
    parser.add_argument('--hessian_smooth_type', type=str, default="constant1e-8",
                        help="hessian smooth type")
    # LOREN
    parser.add_argument('--damping', type=float, default=1.0, help='damping')
    parser.add_argument('--lr_cov', type=float, default=1e-3, help='Learning rate for covariance')
    # evec_a diagnostic
    parser.add_argument('--evec_a_diagnostic', action='store_true',
                        help='Enable evec_a evolution diagnostic logging')
    parser.add_argument('--evec_a_diag_interval', type=int, default=1,
                        help='Log evec_a diagnostic every N steps')
    # AdaSmooth
    parser.add_argument('--beta_init', type=float, default=1.0, help='Initial temperature for AdaSmooth')
    parser.add_argument('--beta_decay', type=float, default=0.05, help='Temperature decay rate for AdaSmooth')
    # ENGINE 
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank for ENGINE')
    parser.add_argument('--ema_gamma', type=float, default=0.9, help='EMA smoothing coefficient for VMS feedback')
    parser.add_argument('--fine_sample_ratio', type=float, default=0.2, help='Ratio of steps with fine residual (1/5 means every 5 steps)')
    parser.add_argument('--no_fine_residual', action='store_true', help='Disable fine grid residual feedback')
    parser.add_argument('--phase1_patience', type=int, default=50, help='Patience for Phase 1 -> Phase 2 transition (default: 50)')
    parser.add_argument('--phase3_epochs', type=int, default=50, help='Number of epochs for Phase 3 fine grid training')
    parser.add_argument('--phase3_lr', type=float, default=None, help='Phase 3 learning rate (default: same as Phase 1)')
    parser.add_argument('--reconstruction_steps', type=int, default=2, help='Post-smoothing steps during reconstruction (2 for Curvature-VMS)')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Phase 1 checkpoint interval (default: 10)')
    parser.add_argument('--phase3_checkpoint_interval', type=int, default=20, help='Phase 3 checkpoint interval (default: 20, saves full model)')
    parser.add_argument('--absorption_interval', type=int, default=0,
                        help='Periodic LoRA absorption every N epochs (0=disabled). Multi V-cycle.')
    parser.add_argument('--vcycle_coarse_epochs', type=int, default=0,
                        help='Coarse grid epochs per V-cycle (0=disabled, use old 3-phase flow)')
    parser.add_argument('--vcycle_fine_epochs', type=int, default=0,
                        help='Fine grid epochs per V-cycle (0=disabled)')
    parser.add_argument('--coarse_isotropic', action='store_true', default=True,
                        help='Use isotropic perturbation on coarse grid (default: True)')
    parser.add_argument('--no_coarse_isotropic', dest='coarse_isotropic', action='store_false',
                        help='Use LOREN-preconditioned perturbation on coarse grid (like V-cycle baseline)')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Custom checkpoint directory (default: {results}/{model}/{task}/EVVMS_checkpoints)')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch number to resume from (for logging)')
    # v1.13 fixes
    parser.add_argument('--tau_min', type=float, default=0.5, help='Minimum τ value to prevent VMS vanishing (v1.13)')
    parser.add_argument('--tau_max', type=float, default=1.0, help='Maximum τ value (for adaptive scheme)')
    parser.add_argument('--tau_scheme', type=str, default='adaptive', choices=['fixed', 'adaptive', 'ema'],
                        help='τ computation scheme: fixed (constant tau_min), adaptive (curvature discrepancy), ema (smoothed adaptive)')
    parser.add_argument('--tau_warmup_epochs', type=int, default=0,
                        help='τ warmup epochs: τ_effective = τ × min(1, epoch/warmup_epochs). 0=disabled')
    parser.add_argument('--no_lora_fp32', action='store_true', help='Disable FP32 for LoRA matrices (use model dtype)')
    parser.add_argument('--tau_lr_scale', action='store_true', help='Enable lr-aware τ scaling: τ *= sqrt(lr_ref/lr) when lr > lr_ref')
    parser.add_argument('--vms_lr_ref', type=float, default=1e-5, help='Reference learning rate for lr-aware τ scaling')
    parser.add_argument('--inject_method', type=str, default='continuous', choices=['intermit', 'continuous'],
                        help='VMS injection method: intermit (only when compute_fine_residual runs), continuous (every step using cached EMA)')
    # LoRA basis selection + B-only coarse grid
    parser.add_argument('--basis_method', type=str, default='random',
                        choices=['random', 'oja', 'svd', 'svd_hybrid', 'online_oja', 'reuse'],
                        help='LoRA A basis selection: random, oja, svd (SVD of momentum), svd_hybrid (evec_a + SVD), online_oja')
    parser.add_argument('--svd_momentum_ema', type=float, default=0.0,
                        help='EMA coefficient for SVD momentum accumulation across cycles (0=disabled)')
    parser.add_argument('--oja_basis', action='store_true', help='[DEPRECATED] Use --basis_method oja')
    parser.add_argument('--coarse_b_only', action='store_true', help='B-only coarse grid training (fix A, only learn B)')
    parser.add_argument('--deep_search', action='store_true',
                        help='Deep search mode: no checkpoint reload at C2F/F2C transitions, '
                             'coarse and fine phases continue from current state instead of best')
    parser.add_argument('--no_checkpoint', action='store_true',
                        help='Disable all checkpoint saving (no best_coarse/best_fine/latest/vcycle_state). '
                             'Fastest mode, but resume is impossible after interruption')
    parser.add_argument('--oja_lr', type=float, default=0.1, help='Oja learning rate (decays as eta_0/(1+0.1*cycle))')
    # FAS residual correction at restriction
    parser.add_argument('--fas_correction', action='store_true', help='Enable FAS residual correction (B warm-start) at restriction')
    parser.add_argument('--fas_alpha', type=float, default=0.5, help='Momentum scaling for B warm-start at restriction')
    parser.add_argument('--fas_w0_alpha', type=float, default=0.0, help='W0 gradient correction step size at restriction (0=disable)')
    # Curvature multigrid cycle (evec_a ↔ evec_a_B/B bidirectional transfer)
    parser.add_argument('--curvature_multigrid', action='store_true',
                        help='Enable curvature multigrid: warm-start evec_a_B/B from evec_a at restriction, '
                             'and blend evec_a_B/B back into evec_a at prolongation')
    parser.add_argument('--curvature_mg_beta', type=float, default=0.3,
                        help='EMA blend coefficient for curvature prolongation (evec_a_B/B → evec_a)')
    # FASLOREN
    parser.add_argument('--fas_coarse_epochs', type=int, default=20, help='Coarse grid epochs per V-cycle')
    parser.add_argument('--fas_fine_epochs', type=int, default=10, help='Fine grid epochs per V-cycle')
    parser.add_argument('--fas_coarse_lr', type=float, default=1e-4, help='Coarse grid (LoRA) learning rate')
    parser.add_argument('--fas_fine_lr', type=float, default=1e-5, help='Fine grid (LOREN) learning rate')
    # HAM2SLOREN (adjoint per-layer learning rate)
    parser.add_argument('--adjoint_mode', type=str, default='fast', choices=['none', 'fast'],
                        help='Adjoint mode: none (no adjoint, baseline), fast (constant terminal condition)')
    parser.add_argument('--adjoint_N', type=int, default=8, help='Adjoint Phase A: normal training steps')
    parser.add_argument('--adjoint_T', type=int, default=8, help='Adjoint Phase B: frozen-alpha steps (recorded)')
    parser.add_argument('--lr_alpha', type=float, default=0.01, help='Learning rate for per-layer alpha update')
    parser.add_argument('--alpha_clip_min', type=float, default=0.1, help='Minimum alpha clipping value')
    parser.add_argument('--alpha_clip_max', type=float, default=10.0, help='Maximum alpha clipping value')
    # LHAM2SLOREN (per-type alpha + Adam for LoRA A)
    parser.add_argument('--adam_lora_B', action='store_true', help='Use Adam optimizer for LoRA A matrix')
    parser.add_argument('--adam_lora_B_beta2', type=float, default=0.999, help='Adam beta2 for LoRA A second moment')
    parser.add_argument('--adam_lora_B_eps', type=float, default=1e-8, help='Adam epsilon for LoRA A')
    parser.add_argument('--alpha_momentum_beta', type=float, default=0.0, help='Alpha momentum beta (0=no momentum)')
    parser.add_argument('--reset_alpha_per_cycle', action='store_true', help='Reset alpha to 1.0 at each V-cycle boundary')
    parser.add_argument('--alpha_inherit', action='store_true', help='Adapt alpha from per-type gradient norms (inherited across V-cycles)')
    # TA-MIGSS (Terrain-Adaptive MIGSS)
    parser.add_argument('--nq_lr_coupling', action='store_true',
                        help='Scale ZO gradient by sqrt((nq-1)/(nq_ref-1)) to compensate step shrinkage at high nq')
    parser.add_argument('--nq_ref', type=int, default=4, help='Reference nq for nq-lr coupling (default: 4)')
    parser.add_argument('--auto_fine_lr', action='store_true',
                        help='Auto-compute fine lr from eta_C/(c*sqrt(h/2r)) based on rank and hidden dim')
    parser.add_argument('--auto_fine_lr_c', type=float, default=0.6,
                        help='Safety coefficient c for auto fine lr (default: 0.6)')
    parser.add_argument('--adaptive_patience', action='store_true',
                        help='P2: End coarse phase when val_acc improvement velocity drops below threshold')
    parser.add_argument('--adaptive_patience_threshold', type=float, default=0.05,
                        help='P2: Min val_acc velocity (pp/epoch) to keep coarse phase running')
    parser.add_argument('--adaptive_patience_min_epochs', type=int, default=8,
                        help='P2: Minimum coarse epochs before adaptive patience can trigger')
    parser.add_argument('--sigma_adaptive_lr', action='store_true',
                        help='Modulate lr by clip(sigma_ema/sigma_f, 1/3, 3) based on ZO query variance')
    # P5: Cross-V-cycle lr decay
    parser.add_argument('--lr_decay', action='store_true',
                        help='Enable cross-V-cycle lr decay when no global improvement')
    parser.add_argument('--lr_decay_patience', type=int, default=3,
                        help='V-cycles without improvement before lr decay')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                        help='Multiply lr by this factor on decay')
    parser.add_argument('--lr_decay_min', type=float, default=1e-7,
                        help='Minimum lr floor')
    # P6: Val_acc velocity-based lr decay (legacy)
    parser.add_argument('--acc_velocity_decay', action='store_true',
                        help='P6: Decay lr when val_acc improvement velocity drops')
    # P5-new: Exploration Boost — boost lr on plateau instead of decay
    parser.add_argument('--exploration_boost', action='store_true',
                        help='P5-new: Boost lr on plateau (exploration-first strategy)')
    parser.add_argument('--boost_factor', type=float, default=1.5,
                        help='P5-new: Multiply lr by this factor on plateau')
    parser.add_argument('--boost_max_ratio', type=float, default=3.0,
                        help='P5-new: Maximum lr boost ratio (lr <= initial_lr * ratio)')
    parser.add_argument('--boost_max_count', type=int, default=3,
                        help='P5-new: Max consecutive boosts before switching to convergence')
    # Auto Warm Restart (AWR) — auto-detect stall and warm restart from best checkpoint
    parser.add_argument('--auto_warm_restart', action='store_true',
                        help='Enable automatic warm restart when training stalls')
    parser.add_argument('--warm_restart_patience', type=int, default=3,
                        help='AWR: V-cycles without global best improvement before warm restart')
    parser.add_argument('--max_warm_restarts', type=int, default=3,
                        help='AWR: Maximum number of warm restarts')
    parser.add_argument('--warm_restart_lr_decay', action='store_true',
                        help='AWR: Decay lr on warm restart')
    parser.add_argument('--warm_restart_lr_factor', type=float, default=0.5,
                        help='AWR: lr multiply factor on warm restart')
    parser.add_argument('--warm_restart_lr_decay_after', type=int, default=1,
                        help='AWR: Start decaying lr after this many warm restarts (0=always)')
    parser.add_argument('--warm_restart_min_gain', type=float, default=0.0,
                        help='AWR: Minimum accuracy gain (pp) per V-cycle to count as improvement (default: 0=any improvement)')
    # Epoch-level lr decay (ReduceLROnPlateau within V-cycle phases)
    parser.add_argument('--epoch_lr_decay', action='store_true',
                        help='Enable epoch-level lr decay within coarse/fine phases')
    parser.add_argument('--epoch_lr_patience', type=int, default=5,
                        help='Epoch-level lr decay: epochs without improvement before decay (counted after warmup)')
    parser.add_argument('--epoch_lr_factor', type=float, default=0.5,
                        help='Epoch-level lr decay: multiply lr by this factor on plateau')
    parser.add_argument('--epoch_lr_min_ratio', type=float, default=0.1,
                        help='Epoch-level lr decay: min lr as ratio of phase-start lr (floor)')
    parser.add_argument('--epoch_lr_warmup', type=int, default=3,
                        help='Epoch-level lr decay: skip first N epochs of each phase (grid-switch settling)')
    # Terrain Sensing (per-step fvar-based lr_mult, inspired by TA-LOREN)
    parser.add_argument('--terrain_sensing', action='store_true',
                        help='Enable per-step terrain sensing for lr modulation')
    parser.add_argument('--ta_boost', type=float, default=0.005,
                        help='Terrain sensing: lr boost rate on flat terrain')
    parser.add_argument('--ta_brake', type=float, default=0.02,
                        help='Terrain sensing: lr brake rate on steep terrain')
    parser.add_argument('--ta_max_mult', type=float, default=2.0,
                        help='Terrain sensing: max lr multiplier')
    parser.add_argument('--ta_warmup', type=int, default=20,
                        help='Terrain sensing: warmup steps for threshold calibration')
    # CT-LOREN (Curvature Transfer)
    parser.add_argument('--curvature_transfer', action='store_true', help='Enable CT-LOREN: transfer full-space curvature to LoRA')
    parser.add_argument('--ct_interval', type=int, default=5, help='CT probe interval (every K steps)')
    parser.add_argument('--ct_beta', type=float, default=0.2, help='CT mixing coefficient (0=no CT, 1=full replace)')
    # Per-row preconditioner
    parser.add_argument('--perrow_precond', action='store_true', help='Per-row independent preconditioner (each row of A/B has own curvature)')
    # Adam for curvature ODE
    parser.add_argument('--adam_cov', action='store_true', help='Use Adam optimizer for curvature ODE (adaptive lr_cov)')
    parser.add_argument('--adam_cov_beta1', type=float, default=0.9, help='Adam beta1 for curvature ODE')
    parser.add_argument('--adam_cov_beta2', type=float, default=0.999, help='Adam beta2 for curvature ODE')
    # lr_cov schedule
    parser.add_argument('--lr_cov_schedule', type=str, default='constant', choices=['constant', 'warmup_cosine'],
                        help='lr_cov schedule: constant (default), warmup_cosine (warmup then cosine decay)')
    parser.add_argument('--lr_cov_warmup_epochs', type=int, default=20, help='lr_cov warmup epochs')
    parser.add_argument('--lr_cov_min', type=float, default=None, help='Minimum lr_cov for decay (default: 0.1 * lr_cov)')

    args = parser.parse_args()
    return args


class _MultiLogger:
    """Thin wrapper that fans out log_metrics() to multiple Lightning loggers."""
    def __init__(self, *loggers):
        self._loggers = loggers

    def log_metrics(self, metrics, step=None):
        for lg in self._loggers:
            lg.log_metrics(metrics, step=step)

    def __bool__(self):
        return True


def setup_logger():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    logfile = os.path.join(args.output_dir, f'log_{timestamp}.txt')
    logger = get_logger(logfile, level="DEBUG")
    get_logger(level='INFO')

    """
    The default of training_args.log_level is passive, so we set log level at info here
    to have that default.
    """
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    return logger


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args = parse_arguments()
    args.hf_token = os.getenv('HF_TOKEN')

    if args.init_seed is not None:
        np.random.seed(args.init_seed)

    trimmed_model_name = args.model_name.split('/')[-1]
    num_steps = math.ceil(args.samplesize / args.batchsize) * args.epochs
    args.run_name = f'{args.algorithm}_lr{args.lr:.0e}_nq-{args.num_queries}'

    if args.algorithm == 'ZOSVRG':
        args.run_name += f'_lr_mb_{args.lr_mezosvrg_mb:.0e}_q_{args.q:.0e}_anneal_{args.anneal:.0e}'
    elif args.algorithm == 'ZOAdam':
        args.run_name += f'_eps_{args.eps:.0e}'
        if args.low_bit_adam == 4 or args.low_bit_adam == 8:
            args.run_name += f'_{args.low_bit_adam}bit'
    elif args.algorithm == 'LOZO':
        args.run_name += f'_rank_{args.rank}_step-interval_{args.step_interval}'
    elif args.algorithm == 'HiZOO':
        args.run_name += f'_hessian_smooth-{args.hessian_smooth_type}'
    elif args.algorithm == 'LOREN':
        args.run_name += f'_damp-{args.damping:.0e}_covlr{args.lr_cov:.0e}'
    elif args.algorithm == 'AdaSmooth':
        args.run_name += f'_beta{args.beta_init:.0e}_decay{args.beta_decay:.0e}_rank{args.rank}'
    elif args.algorithm in ('ENGINE'):
        # Backward compat: --oja_basis maps to --basis_method oja
        if getattr(args, 'oja_basis', False) and args.basis_method == 'random':
            args.basis_method = 'oja'
        args.run_name += f'_damp-{args.damping:.0e}_covlr{args.lr_cov:.0e}_rank{args.lora_rank}'
        if args.basis_method != 'random':
            args.run_name += f'_basis-{args.basis_method}'
        if getattr(args, 'coarse_b_only', False):
            args.run_name += '_bonly'
        if getattr(args, 'fas_correction', False):
            args.run_name += f'_fas-a{args.fas_alpha}'
            if args.fas_w0_alpha > 0:
                args.run_name += f'_w0a{args.fas_w0_alpha}'

    args.run_name += f'_bsz{args.batchsize}_steps{num_steps}'
    args.output_dir = os.path.join(args.results, trimmed_model_name, args.task, args.run_name)

    if args.logging == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger

        writer = TensorBoardLogger(args.output_dir)
    elif args.logging == 'csv':
        from lightning.pytorch.loggers import CSVLogger

        writer = CSVLogger(args.output_dir,
                           flush_logs_every_n_steps=10)
    elif args.logging in ('wandb', 'wandb+csv'):
        os.environ.setdefault('WANDB_MODE', 'offline')
        from lightning.pytorch.loggers import WandbLogger, CSVLogger
        safe_config = {k: v for k, v in vars(args).items() if k != 'hf_token'}
        wandb_run_name = f'{trimmed_model_name}_{args.task}_{args.run_name}'
        wandb_writer = WandbLogger(
            project="ENGINE",
            name=wandb_run_name,
            save_dir=args.output_dir,
            config=safe_config,
        )
        if args.logging == 'wandb+csv':
            csv_writer = CSVLogger(args.output_dir,
                                   flush_logs_every_n_steps=10)
            writer = _MultiLogger(wandb_writer, csv_writer)
        else:
            writer = wandb_writer
    os.makedirs(args.output_dir, exist_ok=True)

    # logging setup
    logger = setup_logger()
    logger.debug(f"Training Arguments {args}")

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    dm = get_data_module(args)
    transformer = get_model(args, dm)

    if 'FO' in args.algorithm:
        finetune_FO(args, device, dm, transformer, writer)
    elif args.algorithm in ('ENGINE'):
        finetune_EVVMS(args, device, dm, transformer, writer)
    else:
        finetune_ZO(args, device, dm, transformer, writer)
