"""
EV-VMS Training Loop

Three-phase training:
- Phase 1: Coarse grid LoRA training with VMS feedback
- Phase 2: Fine grid reconstruction from best checkpoint
- Phase 3: Fine grid LOREN training
"""
import json
import os
import random
import shutil
import time
import numpy as np
import torch
import lightning as L

from common.logging import logger as logging


def _save_vcycle_state(checkpoint_dir, cycle, phase, epoch_in_phase,
                       global_epoch, global_best_acc, global_best_epoch,
                       coarse_best_acc, coarse_best_epoch,
                       fine_best_acc, fine_best_epoch, **extra):
    """Save V-cycle progress for restart (overwrites, ~200 bytes)."""
    state = {
        'cycle': cycle, 'phase': phase,
        'epoch_in_phase': epoch_in_phase,
        'global_epoch': global_epoch,
        'global_best_acc': global_best_acc,
        'global_best_epoch': global_best_epoch,
        'coarse_best_acc': coarse_best_acc,
        'coarse_best_epoch': coarse_best_epoch,
        'fine_best_acc': fine_best_acc,
        'fine_best_epoch': fine_best_epoch,
    }
    state.update(extra)
    with open(os.path.join(checkpoint_dir, 'vcycle_state.json'), 'w') as f:
        json.dump(state, f, indent=2)


def _update_global_best_link(global_best_path, target_dir):
    """Make best_global a symlink to best_coarse or best_fine."""
    rel = os.path.relpath(target_dir, os.path.dirname(global_best_path))
    if os.path.islink(global_best_path):
        if os.readlink(global_best_path) == rel:
            return
        os.unlink(global_best_path)
    elif os.path.isdir(global_best_path):
        shutil.rmtree(global_best_path)
    elif os.path.exists(global_best_path):
        os.remove(global_best_path)
    os.symlink(rel, global_best_path)


def _cleanup_shared(shared_dir, checkpoint_dirs):
    """Remove shared files not referenced by any checkpoint symlink."""
    if not os.path.isdir(shared_dir):
        return
    referenced = set()
    for ckpt_dir in checkpoint_dirs:
        if not os.path.isdir(ckpt_dir):
            continue
        for f in os.listdir(ckpt_dir):
            fp = os.path.join(ckpt_dir, f)
            if os.path.islink(fp):
                referenced.add(os.path.realpath(fp))
    for f in os.listdir(shared_dir):
        fp = os.path.join(shared_dir, f)
        if os.path.realpath(fp) not in referenced:
            try:
                os.remove(fp)
                logging.info(f"  Cleaned up unreferenced: shared/{f}")
            except (FileNotFoundError, OSError):
                pass  # Already removed, or NFS busy (.nfs files)


def reset_log_dict(log_dict):
    """Clear metric keys between logging points.

    Keys are removed (not zeroed) so that only freshly-written metrics
    appear in the next log_metrics() call, preventing spurious zeros
    in wandb/CSV (e.g. train_loss=0 at validation steps).
    """
    for k in ('train_loss', 'val_loss', 'val_acc', 'time'):
        log_dict.pop(k, None)


def evaluate(transformer, val_dataloader, device, log_dict):
    """Run validation and return metrics."""
    val_loss_sums, val_correct, total_val_samples = [], 0, 0
    for batch in val_dataloader:
        b = {k: v.to(device) for k, v in batch.items()}
        loss_sum, correct, total_samples = transformer.forward_ZO_val(b)
        val_loss_sums.append(loss_sum.item())
        val_correct += correct
        total_val_samples += total_samples

    val_loss_mean = np.sum(val_loss_sums) / total_val_samples
    val_acc = val_correct / total_val_samples * 100.0

    log_dict['val_loss'] = val_loss_mean
    log_dict['val_acc'] = val_acc

    return val_loss_mean, val_acc


def finetune_EVVMS(args,
                     device: torch.device,
                     dm: L.LightningDataModule,
                     transformer,
                     tb_writer=None):
    """
    EV-VMS three-phase training.

    Phase 1: Coarse grid LoRA training with VMS feedback
    Phase 2: Fine grid reconstruction from best checkpoint
    Phase 3: Fine grid LOREN training
    """
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    log_dict = {}
    kwargs = {'log': log_dict}

    if args.bf16:
        transformer.to(torch.bfloat16)

    model = transformer.model.to(device)

    # Fix ALL random seeds for reproducibility
    # np.random: perturbation seeds in training_step (line 972) and VMS (line 1006)
    # torch: LoRA init via torch.randn in configure_lora_params, DataLoader shuffle
    # random: Python built-in (just in case)
    if args.init_seed is not None:
        np.random.seed(args.init_seed)
        torch.manual_seed(args.init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.init_seed)
        random.seed(args.init_seed)

    # Configure parameters (LOREN params + LoRA params)
    transformer.configure_params()

    # Enable terrain sensing if requested
    if getattr(args, 'terrain_sensing', False) and hasattr(transformer, 'enable_terrain_sensing'):
        transformer.enable_terrain_sensing(
            boost=getattr(args, 'ta_boost', 0.005),
            brake=getattr(args, 'ta_brake', 0.02),
            max_mult=getattr(args, 'ta_max_mult', 2.0),
            warmup=getattr(args, 'ta_warmup', 20),
        )
        logging.info(f"  [TA-IGSS] Terrain sensing enabled: "
                     f"boost={args.ta_boost}, brake={args.ta_brake}, "
                     f"max_mult={args.ta_max_mult}, warmup={args.ta_warmup}")

    transformer.model.eval()

    # Print LoRA info
    total_lora_params = 0
    for name in transformer.lora_B:
        A = transformer.lora_B[name]
        B = transformer.lora_A[name]
        total_lora_params += A.numel() + B.numel()
    logging.info(f"LoRA parameters: {total_lora_params / 1e6:.2f}M (rank={transformer.lora_rank})")
    logging.info(f"EMA gamma: {transformer.ema_gamma}")
    logging.info(f"Fine residual interval: every {transformer.fine_residual_interval} steps")

    # Tracking
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = None
    prev_best_phase1_path = None  # Track previous best checkpoint for cleanup
    first_periodic_saved = False  # Track if first periodic checkpoint (with full model) is saved
    total_forward_passes = 0  # Independent counter: NOT reset by load_from_checkpoint

    # Phase settings
    phase1_patience = getattr(args, 'phase1_patience', 50)  # Patience for Phase 1 → Phase 2 transition
    coarse_patience = getattr(args, 'coarse_patience', 0)   # V-cycle coarse early stopping (0=disabled)
    fine_patience = getattr(args, 'fine_patience', 0)       # V-cycle fine early stopping (0=disabled)
    phase3_epochs = getattr(args, 'phase3_epochs', 50)  # Epochs for Phase 3
    reconstruction_steps = getattr(args, 'reconstruction_steps', 0)  # 0: prolongation is exact, post-smoothing hurts
    checkpoint_interval = getattr(args, 'checkpoint_interval', 10)  # Phase 1 checkpoint interval
    phase3_checkpoint_interval = getattr(args, 'phase3_checkpoint_interval', 20)  # Phase 3 checkpoint interval (full model)
    phase3_lr = getattr(args, 'phase3_lr', None)  # Phase 3 learning rate (None = same as Phase 1)

    # Checkpoint directory
    if getattr(args, 'checkpoint_dir', None):
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = os.path.join(
            args.results,
            args.model_name.replace('/', '_'),
            args.task,
            f"EVVMS_checkpoints"
        )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check V-cycle mode
    vcycle_M = getattr(args, 'vcycle_coarse_epochs', 0)
    vcycle_N = getattr(args, 'vcycle_fine_epochs', 0)

    if vcycle_M > 0 and vcycle_N > 0:
        # ========================================
        # V-Cycle Training Mode
        # ========================================
        num_vcycles = args.epochs // (vcycle_M + vcycle_N)
        if num_vcycles < 1:
            num_vcycles = 1
        total_epoch_budget = args.epochs  # Run until this many epochs are consumed
        phase3_lr = getattr(args, 'phase3_lr', None)

        # P1: Auto fine lr from rank-dependent formula: η_F = η_C / (c * (h/2r)^(1/4))
        auto_fine_lr = getattr(args, 'auto_fine_lr', False)
        if auto_fine_lr and phase3_lr is None:
            h = transformer.lora_rank  # hidden dim per LoRA layer
            r = transformer.lora_rank
            # Get actual hidden dimension from model
            model_config = transformer.model.config
            if hasattr(model_config, 'hidden_size'):
                h = model_config.hidden_size
            elif hasattr(model_config, 'n_embd'):
                h = model_config.n_embd
            elif hasattr(model_config, 'dim'):
                h = model_config.dim
            c = getattr(args, 'auto_fine_lr_c', 0.6)
            import math as _math
            ratio = c * (h / (2.0 * r)) ** 0.25  # 4th-root: milder than sqrt for large h
            ratio = max(ratio, 1.0)
            phase3_lr = args.lr / ratio
            logging.info(f"  [TA-MIGSS] Auto fine lr: η_F = {args.lr} / {ratio:.1f} = {phase3_lr:.2e} "
                         f"(h={h}, r={r}, c={c})")

        # P2: Adaptive patience config
        adaptive_patience = getattr(args, 'adaptive_patience', False)
        adaptive_patience_threshold = getattr(args, 'adaptive_patience_threshold', 1e-4)
        adaptive_patience_min_epochs = getattr(args, 'adaptive_patience_min_epochs', 8)

        # P5: Cross-V-cycle lr decay — decay lr when consecutive V-cycles show no global improvement
        lr_decay_enabled = getattr(args, 'lr_decay', False)
        lr_decay_patience = getattr(args, 'lr_decay_patience', 3)   # V-cycles without improvement before decay
        lr_decay_factor = getattr(args, 'lr_decay_factor', 0.5)     # multiply lr by this factor
        lr_decay_min = getattr(args, 'lr_decay_min', 1e-7)          # minimum lr floor

        # P5-new: Exploration Boost — boost lr on plateau (overrides P5 if enabled)
        exploration_boost = getattr(args, 'exploration_boost', False)
        boost_factor = getattr(args, 'boost_factor', 1.5)
        boost_max_ratio = getattr(args, 'boost_max_ratio', 3.0)
        boost_max_count = getattr(args, 'boost_max_count', 3)
        lr_initial = args.lr  # record initial lr for boost ceiling

        # P6: Val_acc velocity-based lr decay — decay when improvement rate drops
        acc_velocity_decay = getattr(args, 'acc_velocity_decay', False)
        acc_velocity_ema = 0.0       # EMA of per-cycle acc velocity
        acc_velocity_ema_init = False # Whether EMA has been initialized

        # AWR: Auto Warm Restart — detect stall and warm restart from best checkpoint
        auto_warm_restart = getattr(args, 'auto_warm_restart', False)
        warm_restart_patience = getattr(args, 'warm_restart_patience', 3)
        max_warm_restarts = getattr(args, 'max_warm_restarts', 3)
        warm_restart_lr_decay = getattr(args, 'warm_restart_lr_decay', False)
        warm_restart_lr_factor = getattr(args, 'warm_restart_lr_factor', 0.5)
        warm_restart_lr_decay_after = getattr(args, 'warm_restart_lr_decay_after', 1)
        warm_restart_min_gain = getattr(args, 'warm_restart_min_gain', 0.0)
        warm_restart_count = 0
        no_improve_since_restart = 0  # V-cycles without improvement since last warm restart
        awr_best_at_restart = 0.0     # global_best_acc at last AWR reset point

        # Epoch-level lr decay (ReduceLROnPlateau within phases)
        epoch_lr_decay = getattr(args, 'epoch_lr_decay', False)
        epoch_lr_patience = getattr(args, 'epoch_lr_patience', 5)
        epoch_lr_factor = getattr(args, 'epoch_lr_factor', 0.5)
        epoch_lr_min_ratio = getattr(args, 'epoch_lr_min_ratio', 0.1)
        epoch_lr_warmup = getattr(args, 'epoch_lr_warmup', 3)

        # Deep search mode: no checkpoint reload at C2F/F2C
        deep_search = getattr(args, 'deep_search', False)
        no_checkpoint = getattr(args, 'no_checkpoint', False)

        logging.info("=" * 60)
        logging.info("V-Cycle Training Mode")
        logging.info(f"  Coarse epochs per cycle (M): {vcycle_M} (max, with early stopping)")
        logging.info(f"  Fine epochs per cycle (N): {vcycle_N}")
        logging.info(f"  Total epoch budget: {total_epoch_budget}")
        logging.info(f"  Estimated V-cycles: {num_vcycles} (may run more if coarse early-stops)")
        logging.info(f"  Coarse lr: {args.lr}, Fine lr: {phase3_lr if phase3_lr else args.lr}")
        logging.info(f"  LoRA rank: {transformer.lora_rank}")
        logging.info(f"  VMS feedback: {'enabled' if transformer.use_fine_residual else 'disabled'}")
        if adaptive_patience:
            logging.info(f"  [TA-MIGSS] Adaptive patience (val_acc): threshold={adaptive_patience_threshold}pp/ep, "
                         f"min_epochs={adaptive_patience_min_epochs}")
        if lr_decay_enabled:
            logging.info(f"  [TA-MIGSS] Cross-V-cycle lr decay: patience={lr_decay_patience}, "
                         f"factor={lr_decay_factor}, min_lr={lr_decay_min}")
        if exploration_boost:
            logging.info(f"  [TA-MIGSS] P5-new Exploration Boost: factor={boost_factor}, "
                         f"max_ratio={boost_max_ratio}x, max_boosts={boost_max_count}")
        if acc_velocity_decay:
            logging.info(f"  [TA-MIGSS] P6 acc-velocity lr decay: enabled (EMA-based)")
        if auto_warm_restart:
            logging.info(f"  [AWR] Auto Warm Restart: patience={warm_restart_patience} cycles, "
                         f"max_restarts={max_warm_restarts}, min_gain={warm_restart_min_gain}pp, "
                         f"lr_decay={'on (factor=' + str(warm_restart_lr_factor) + ', after=' + str(warm_restart_lr_decay_after) + ')' if warm_restart_lr_decay else 'off'}")
        if epoch_lr_decay:
            logging.info(f"  [EpochLR] Epoch-level lr decay: patience={epoch_lr_patience}ep, "
                         f"factor={epoch_lr_factor}, min_ratio={epoch_lr_min_ratio}, warmup={epoch_lr_warmup}ep")
        coarse_iso = getattr(args, 'coarse_isotropic', True)
        logging.info(f"  Coarse grid perturbation: {'isotropic' if coarse_iso else 'LOREN-preconditioned'}")
        if deep_search:
            logging.info(f"  Deep search: ON (no checkpoint reload at C2F/F2C)")
        if no_checkpoint:
            logging.info(f"  Checkpointing: OFF (no save, resume impossible)")
            if not deep_search:
                logging.warning("  WARNING: --no_checkpoint without --deep_search: "
                                "C2F/F2C will have no best checkpoint to load")
        logging.info("=" * 60)

        global_best_acc = 0.0
        global_best_epoch = 0
        coarse_best_acc = 0.0
        coarse_best_epoch = 0
        fine_best_acc = 0.0
        fine_best_epoch = 0
        global_epoch = 0
        no_improve_cycles = 0  # P5: consecutive V-cycles without global_best improvement
        lr_decay_count = 0     # P5: how many times lr has been decayed
        boost_count = 0        # P5-new: consecutive boosts applied
        explore_phase = "exploration"  # P5-new: "exploration" or "convergence"
        lr_before_boost = args.lr  # P5-new: lr before last boost (for divergence rollback)
        cycle_start_best = 0.0 # P5: global_best_acc at start of current V-cycle

        # Fixed checkpoint paths (overwrite to save disk)
        global_best_path = os.path.join(checkpoint_dir, "best_global")
        coarse_best_path = os.path.join(checkpoint_dir, "best_coarse")
        fine_best_path = os.path.join(checkpoint_dir, "best_fine")
        shared_dir = os.path.join(checkpoint_dir, "shared")
        latest_path = os.path.join(checkpoint_dir, "latest")  # Deep search resume

        from model.glue_transformer_loren import GLUETransformerLOREN

        # ---- V-Cycle Resume Logic ----
        start_cycle = 0
        start_coarse_ep = 0
        start_fine_ep = 0
        resume_phase = None  # None = fresh start

        resume_checkpoint = getattr(args, 'resume_checkpoint', None)
        vcycle_state_path = os.path.join(checkpoint_dir, 'vcycle_state.json')

        if resume_checkpoint and os.path.exists(vcycle_state_path):
            with open(vcycle_state_path) as f:
                vs = json.load(f)
            start_cycle = vs['cycle']
            resume_phase = vs['phase']
            global_epoch = vs['global_epoch']
            global_best_acc = vs['global_best_acc']
            global_best_epoch = vs['global_best_epoch']
            coarse_best_acc = vs.get('coarse_best_acc', 0.0)
            coarse_best_epoch = vs.get('coarse_best_epoch', 0)
            fine_best_acc = vs.get('fine_best_acc', 0.0)
            fine_best_epoch = vs.get('fine_best_epoch', 0)
            # Restore AWR state
            warm_restart_count = vs.get('warm_restart_count', 0)
            no_improve_since_restart = vs.get('no_improve_since_restart', 0)
            awr_best_at_restart = vs.get('awr_best_at_restart', global_best_acc)

            # Load checkpoint for the interrupted phase
            loaded = False
            was_deep_search = vs.get('deep_search', False)

            if was_deep_search and os.path.exists(latest_path):
                # Deep search resume: load latest state (not best)
                transformer.load_from_checkpoint(latest_path)
                if resume_phase == 'coarse':
                    start_coarse_ep = vs['epoch_in_phase'] + 1
                elif resume_phase == 'fine':
                    start_fine_ep = vs['epoch_in_phase'] + 1
                loaded = True
            elif resume_phase == 'coarse' and os.path.exists(coarse_best_path):
                transformer.load_from_checkpoint(coarse_best_path)
                start_coarse_ep = vs['epoch_in_phase'] + 1
                loaded = True
            elif resume_phase == 'fine' and os.path.exists(fine_best_path):
                transformer.load_from_checkpoint(fine_best_path)
                start_fine_ep = vs['epoch_in_phase'] + 1
                loaded = True

            if not loaded and os.path.exists(global_best_path):
                transformer.load_from_checkpoint(global_best_path)
                start_coarse_ep = vs['epoch_in_phase'] + 1
                loaded = True

            if loaded:
                logging.info("=" * 60)
                logging.info("V-Cycle Resumed from checkpoint")
                logging.info(f"  Cycle: {start_cycle + 1}/{num_vcycles}, Phase: {resume_phase}")
                logging.info(f"  Resume from epoch_in_phase: {vs['epoch_in_phase'] + 1}")
                logging.info(f"  Global epoch: {global_epoch}, Best: {global_best_acc:.3f}%")
                logging.info("=" * 60)
            else:
                logging.warning("Resume requested but no checkpoint found, starting fresh")
                resume_phase = None
        elif resume_checkpoint:
            # No vcycle_state.json — load checkpoint and start fresh V-cycles
            if os.path.isdir(resume_checkpoint):
                transformer.load_from_checkpoint(resume_checkpoint)
                # Check for cold restart info (cumulative epoch/best tracking)
                cold_info_path = os.path.join(resume_checkpoint, 'cold_restart_info.json')
                if os.path.exists(cold_info_path):
                    with open(cold_info_path) as f:
                        cold_info = json.load(f)
                    global_epoch = cold_info.get('global_epoch', 0)
                    global_best_acc = cold_info.get('global_best_acc', 0.0)
                    global_best_epoch = cold_info.get('global_best_epoch', 0)
                    total_epoch_budget = global_epoch + args.epochs
                    logging.info("=" * 60)
                    logging.info("V-Cycle Cold Restart from checkpoint")
                    logging.info(f"  Loaded: {resume_checkpoint}")
                    logging.info(f"  Cumulative epoch: {global_epoch}, Best: {global_best_acc:.3f}% (ep {global_best_epoch})")
                    logging.info(f"  Epoch budget: {global_epoch} + {args.epochs} = {total_epoch_budget}")
                    logging.info(f"  Starting {num_vcycles} fresh V-cycles")
                    logging.info("=" * 60)
                else:
                    logging.info("=" * 60)
                    logging.info("V-Cycle Fresh Start from checkpoint")
                    logging.info(f"  Loaded: {resume_checkpoint}")
                    logging.info(f"  Starting {num_vcycles} fresh V-cycles")
                    logging.info("=" * 60)
            else:
                logging.warning(f"Resume requested but {resume_checkpoint} not found, starting fresh")

        cycle = start_cycle - 1
        while global_epoch < total_epoch_budget:
            cycle += 1
            cycle_start_best = global_best_acc  # P5: record best at cycle start
            cycle_start_epoch = global_epoch      # P6: record epoch at cycle start

            # ---- Coarse Grid Phase (M epochs) ----
            skip_coarse = (cycle == start_cycle and resume_phase == 'fine')
            coarse_ep_start = start_coarse_ep if (cycle == start_cycle and resume_phase == 'coarse') else 0

            if not skip_coarse:
                logging.info("=" * 60)
                logging.info(f"[V-cycle {cycle+1}] Coarse Grid (max {vcycle_M} epochs)")
                if coarse_ep_start > 0:
                    logging.info(f"  Resuming from epoch {coarse_ep_start + 1}")
                logging.info("=" * 60)

                transformer.set_phase('coarse')

                # §4.2/4.3 fix: reset coarse_best per cycle so each cycle saves
                # its own best checkpoint for C2F prolongation, instead of reusing
                # a stale historical best that never updates
                if cycle > 0 and not (cycle == start_cycle and resume_phase == 'coarse'):
                    logging.info(f"  Per-cycle coarse_best reset: {coarse_best_acc:.3f}% -> 0.0%")
                    coarse_best_acc = 0.0
                    coarse_best_epoch = 0
                    transformer.restriction_to_coarse()

                # Restore coarse learning rate
                transformer.state.learning_rate = args.lr
                # Reset terrain sensing for new grid
                if hasattr(transformer, 'reset_terrain_sensing'):
                    transformer.reset_terrain_sensing()
                    if hasattr(transformer, '_terrain_enabled') and transformer._terrain_enabled:
                        transformer._terrain_lr_base = args.lr

                coarse_phase_best = 0.0   # Best val_acc within this coarse phase
                coarse_phase_patience = 0  # Patience counter for coarse early stopping
                epoch_lr_no_improve_C = 0  # Epoch-level lr decay counter for coarse
                effective_lr_C = args.lr   # Current effective coarse lr (may be epoch-decayed)
                # P2: Val_acc velocity tracking (replaces val_loss EMA)
                coarse_phase_start_acc = global_best_acc  # acc at start of this coarse phase

                for ep in range(coarse_ep_start, vcycle_M):
                    epoch_start_time = time.time()
                    if global_epoch >= total_epoch_budget:
                        break
                    global_epoch += 1
                    log_dict['epoch'] = global_epoch
                    log_dict['phase'] = 1

                    transformer.set_epoch(global_epoch)

                    # Validation
                    if global_epoch % args.eval_every == 0:
                        reset_log_dict(log_dict)
                        val_loss_mean, val_acc = evaluate(transformer, val_dataloader, device, log_dict)

                        # Coarse best (historical, not reset across cycles)
                        if val_acc > coarse_best_acc:
                            coarse_best_acc = val_acc
                            coarse_best_epoch = global_epoch
                            if not no_checkpoint:
                                transformer.save_checkpoint(coarse_best_path, save_full_model=True,
                                                            shared_dir=shared_dir, epoch_tag=global_epoch)
                            # Update global best symlink if needed
                            if val_acc > global_best_acc:
                                global_best_acc = val_acc
                                global_best_epoch = global_epoch
                                if not no_checkpoint:
                                    _update_global_best_link(global_best_path, coarse_best_path)
                                logging.info(f"  -> New global best! {val_acc:.3f}% (ep {global_epoch})")

                        if not no_checkpoint:
                            _cleanup_shared(shared_dir, [coarse_best_path, fine_best_path])

                        # Per-phase early stopping: track best within this coarse phase
                        if val_acc > coarse_phase_best:
                            coarse_phase_best = val_acc
                            coarse_phase_patience = 0
                            epoch_lr_no_improve_C = 0
                        else:
                            coarse_phase_patience += 1
                            # Only count for epoch_lr after warmup (grid-switch settling)
                            if ep >= epoch_lr_warmup:
                                epoch_lr_no_improve_C += 1

                        # Epoch-level lr decay for coarse phase
                        if (epoch_lr_decay and ep >= epoch_lr_warmup
                                and epoch_lr_no_improve_C >= epoch_lr_patience):
                            lr_floor = args.lr * epoch_lr_min_ratio
                            if effective_lr_C > lr_floor:
                                old_lr = effective_lr_C
                                effective_lr_C = max(effective_lr_C * epoch_lr_factor, lr_floor)
                                transformer.state.learning_rate = effective_lr_C
                                logging.info(f"  [EpochLR] Coarse lr decay: {old_lr:.2e} -> {effective_lr_C:.2e} "
                                             f"(no improve for {epoch_lr_no_improve_C}ep, floor={lr_floor:.2e})")
                                epoch_lr_no_improve_C = 0

                        log_dict["best_val_acc"] = global_best_acc
                        if tb_writer:
                            tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

                        # Terrain macro signal: update val_acc for acc_boost (coarse grid)
                        if hasattr(transformer, 'update_val_signal'):
                            transformer.update_val_signal(val_acc)

                        _fine_lr_display = phase3_lr if phase3_lr else args.lr
                        _coarse_lr_display = effective_lr_C if epoch_lr_decay else args.lr
                        logging.info(f"[V{cycle+1}][Coarse Ep {ep+1}/{vcycle_M}] val_loss: {val_loss_mean:.5f}, "
                                    f"val_acc: {val_acc:.3f}%, best: {global_best_acc:.3f}% (g.ep {global_best_epoch}), "
                                    f"coarse_best: {coarse_best_acc:.3f}% (ep {coarse_best_epoch}), "
                                    f"lr_C: {_coarse_lr_display:.2e}, lr_F: {_fine_lr_display:.2e}")

                        if coarse_patience > 0 and coarse_phase_patience >= coarse_patience:
                            logging.info(f"  [V{cycle+1}] Coarse early stop: no improvement for {coarse_patience} epochs "
                                         f"(phase best: {coarse_phase_best:.3f}%), advancing to fine grid")
                            break

                        # P2: Val_acc-based adaptive patience — stop when acc improvement rate drops
                        if adaptive_patience and ep + 1 >= adaptive_patience_min_epochs:
                            acc_gain = coarse_phase_best - coarse_phase_start_acc
                            acc_velocity = acc_gain / (ep + 1)  # acc improvement per epoch
                            if acc_velocity < adaptive_patience_threshold:
                                logging.info(f"  [V{cycle+1}] Adaptive patience: acc_velocity={acc_velocity:.4f}pp/ep "
                                             f"(gain={acc_gain:.3f}pp in {ep+1}ep) "
                                             f"< threshold={adaptive_patience_threshold}, advancing to fine grid")
                                break

                    # Training
                    transformer.model.train()
                    train_loss = 0.0
                    for batch_idx, batch in enumerate(train_dataloader):
                        reset_log_dict(log_dict)
                        b = {k: v.to(device) for k, v in batch.items()}
                        transformer.training_step(model, b, tb_writer=tb_writer, **kwargs)
                        total_forward_passes += transformer.state.query[-1]
                        log_dict['num_forward_passes'] = total_forward_passes
                        train_loss += log_dict.get('train_loss', 0.0)
                        if tb_writer:
                            tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

                    alloc_mem = transformer.measure_memory_usage()
                    epoch_time = time.time() - epoch_start_time
                    log_dict['memory_gb'] = alloc_mem
                    log_dict['epoch_time_sec'] = epoch_time
                    if tb_writer:
                        tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

                    train_loss /= len(train_dataloader)
                    vms_info = ''
                    if transformer.use_fine_residual and 'vms_c_B_mean' in log_dict:
                        vms_info = (f", VMS: c_B={log_dict.get('vms_c_B_mean', 0):.4e}"
                                   f" c_A={log_dict.get('vms_c_A_mean', 0):.4e}"
                                   f" c_f={log_dict.get('vms_c_f_raw_mean', 0):.4e}"
                                   f" c_f'={log_dict.get('vms_c_f_prime_raw_mean', 0):.4e}"
                                   f" τ={log_dict.get('vms_tau_mean', 0):.3f}")
                    logging.info(f'[V{cycle+1}][Coarse Ep {ep+1}/{vcycle_M}] train_loss: {train_loss:.5f}, mem: {alloc_mem:.2f}GB{vms_info}')

                    # Save V-cycle progress for restart
                    if not no_checkpoint:
                        _save_vcycle_state(checkpoint_dir, cycle, 'coarse', ep,
                                          global_epoch, global_best_acc, global_best_epoch,
                                          coarse_best_acc, coarse_best_epoch,
                                          fine_best_acc, fine_best_epoch,
                                          warm_restart_count=warm_restart_count,
                                          no_improve_since_restart=no_improve_since_restart,
                                          awr_best_at_restart=awr_best_at_restart,
                                          deep_search=deep_search)

                        # Deep search: save latest checkpoint every epoch for resume
                        if deep_search:
                            # Remove stale fine_state.pt so load_from_checkpoint detects coarse format
                            _stale = os.path.join(latest_path, 'fine_state.pt')
                            if os.path.exists(_stale):
                                os.remove(_stale)
                            transformer.save_checkpoint(latest_path, save_full_model=True)

                if not deep_search:
                    if os.path.exists(coarse_best_path):
                        logging.info(f"  Loading coarse best (acc={coarse_best_acc:.3f}%, ep {coarse_best_epoch}) for prolongation")
                        transformer.load_from_checkpoint(coarse_best_path)
                else:
                    logging.info(f"  Deep search: continuing from current state (no checkpoint reload)")

                # ---- Prolongation (Coarse -> Fine) ----
                logging.info("-" * 40)
                transformer.prolongation_to_fine()

            # ---- Fine Grid Phase (N epochs) ----
            fine_ep_start = start_fine_ep if (cycle == start_cycle and resume_phase == 'fine') else 0

            logging.info("=" * 60)
            logging.info(f"[V-cycle {cycle+1}] Fine Grid ({vcycle_N} epochs)")
            if fine_ep_start > 0:
                logging.info(f"  Resuming from epoch {fine_ep_start + 1}")
            logging.info("=" * 60)

            transformer.set_phase('fine')

            # §4.2/4.3 fix: reset fine_best per cycle (same rationale as coarse_best reset)
            # so each cycle saves its own best fine checkpoint for F2C restriction,
            # instead of reusing a stale historical best that never updates
            if cycle > 0 and not (cycle == start_cycle and resume_phase == 'fine'):
                logging.info(f"  Per-cycle fine_best reset: {fine_best_acc:.3f}% -> 0.0%")
                fine_best_acc = 0.0
                fine_best_epoch = 0

            if phase3_lr is not None:
                transformer.state.learning_rate = phase3_lr
            # Reset terrain sensing for fine grid
            if hasattr(transformer, 'reset_terrain_sensing'):
                transformer.reset_terrain_sensing()
                if hasattr(transformer, '_terrain_enabled') and transformer._terrain_enabled:
                    transformer._terrain_lr_base = phase3_lr if phase3_lr else transformer.state.learning_rate

            fine_phase_best = 0.0   # Best val_acc within this fine phase
            fine_phase_patience = 0  # Patience counter for fine early stopping
            epoch_lr_no_improve_F = 0  # Epoch-level lr decay counter for fine
            effective_lr_F = phase3_lr if phase3_lr else args.lr  # Current effective fine lr

            for ep in range(fine_ep_start, vcycle_N):
                epoch_start_time = time.time()
                if global_epoch >= total_epoch_budget:
                    break
                global_epoch += 1
                log_dict['epoch'] = global_epoch
                log_dict['phase'] = 3

                transformer.set_epoch(global_epoch)

                # Validation
                if global_epoch % args.eval_every == 0:
                    reset_log_dict(log_dict)
                    val_loss_mean, val_acc = evaluate(transformer, val_dataloader, device, log_dict)

                    # Fine best (historical, not reset across cycles)
                    if val_acc > fine_best_acc:
                        fine_best_acc = val_acc
                        fine_best_epoch = global_epoch
                        if not no_checkpoint:
                            transformer.save_checkpoint(fine_best_path,
                                                        shared_dir=shared_dir, epoch_tag=global_epoch)
                        # Update global best symlink if needed
                        if val_acc > global_best_acc:
                            global_best_acc = val_acc
                            global_best_epoch = global_epoch
                            if not no_checkpoint:
                                _update_global_best_link(global_best_path, fine_best_path)
                            logging.info(f"  -> New global best! {val_acc:.3f}% (ep {global_epoch})")

                    if not no_checkpoint:
                        _cleanup_shared(shared_dir, [coarse_best_path, fine_best_path])

                    # Per-phase early stopping for fine grid
                    if val_acc > fine_phase_best:
                        fine_phase_best = val_acc
                        fine_phase_patience = 0
                        epoch_lr_no_improve_F = 0
                    else:
                        fine_phase_patience += 1
                        # Only count for epoch_lr after warmup (grid-switch settling)
                        if ep >= epoch_lr_warmup:
                            epoch_lr_no_improve_F += 1

                    # Epoch-level lr decay for fine phase
                    if (epoch_lr_decay and ep >= epoch_lr_warmup
                            and epoch_lr_no_improve_F >= epoch_lr_patience):
                        base_fine_lr = phase3_lr if phase3_lr else args.lr
                        lr_floor = base_fine_lr * epoch_lr_min_ratio
                        if effective_lr_F > lr_floor:
                            old_lr_F = effective_lr_F
                            effective_lr_F = max(effective_lr_F * epoch_lr_factor, lr_floor)
                            transformer.state.learning_rate = effective_lr_F
                            logging.info(f"  [EpochLR] Fine lr decay: {old_lr_F:.2e} -> {effective_lr_F:.2e} "
                                         f"(no improve for {epoch_lr_no_improve_F}ep, floor={lr_floor:.2e})")
                            epoch_lr_no_improve_F = 0

                    log_dict["best_val_acc"] = global_best_acc
                    if tb_writer:
                        tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

                    # Terrain macro signal: update val_acc for acc_boost (fine grid)
                    if hasattr(transformer, 'update_val_signal'):
                        transformer.update_val_signal(val_acc)

                    _fine_lr_display = effective_lr_F if epoch_lr_decay else (phase3_lr if phase3_lr else args.lr)
                    logging.info(f"[V{cycle+1}][Fine Ep {ep+1}/{vcycle_N}] val_loss: {val_loss_mean:.5f}, "
                                f"val_acc: {val_acc:.3f}%, best: {global_best_acc:.3f}% (g.ep {global_best_epoch}), "
                                f"fine_best: {fine_best_acc:.3f}% (ep {fine_best_epoch}), "
                                f"lr_C: {args.lr:.2e}, lr_F: {_fine_lr_display:.2e}")

                    if fine_patience > 0 and fine_phase_patience >= fine_patience:
                        logging.info(f"  [V{cycle+1}] Fine early stop: no improvement for {fine_patience} epochs "
                                     f"(phase best: {fine_phase_best:.3f}%), advancing to next V-cycle")
                        break

                # Fine grid training (LOREN full-param step + terrain sensing)
                transformer.model.train()
                train_loss = 0.0
                for batch_idx, batch in enumerate(train_dataloader):
                    reset_log_dict(log_dict)
                    b = {k: v.to(device) for k, v in batch.items()}
                    GLUETransformerLOREN.training_step(transformer, model, b, tb_writer=tb_writer, **kwargs)
                    # Terrain sensing on fine grid (fvals stored by LOREN.training_step)
                    if hasattr(transformer, '_terrain_enabled') and transformer._terrain_enabled:
                        if hasattr(transformer, '_last_fvals') and hasattr(transformer, '_update_terrain'):
                            transformer._update_terrain(transformer._last_fvals, transformer._last_fmean)
                            log_dict['lr_mult'] = transformer.get_terrain_lr_mult()
                    total_forward_passes += transformer.state.query[-1]
                    log_dict['num_forward_passes'] = total_forward_passes
                    train_loss += log_dict.get('train_loss', 0.0)
                    if tb_writer:
                        tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

                alloc_mem = transformer.measure_memory_usage()
                epoch_time = time.time() - epoch_start_time
                log_dict['memory_gb'] = alloc_mem
                log_dict['epoch_time_sec'] = epoch_time
                if tb_writer:
                    tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

                train_loss /= len(train_dataloader)
                logging.info(f'[V{cycle+1}][Fine Ep {ep+1}/{vcycle_N}] train_loss: {train_loss:.5f}, mem: {alloc_mem:.2f}GB')

                # Save V-cycle progress for restart
                if not no_checkpoint:
                    _save_vcycle_state(checkpoint_dir, cycle, 'fine', ep,
                                      global_epoch, global_best_acc, global_best_epoch,
                                      coarse_best_acc, coarse_best_epoch,
                                      fine_best_acc, fine_best_epoch,
                                      warm_restart_count=warm_restart_count,
                                      no_improve_since_restart=no_improve_since_restart,
                                      awr_best_at_restart=awr_best_at_restart,
                                      deep_search=deep_search)

                    # Deep search: save latest checkpoint every epoch for resume
                    if deep_search:
                        # Remove stale coarse files so load_from_checkpoint detects fine format
                        for _stale_name in ('lora_state.pt', 'params_1d.pt'):
                            _stale = os.path.join(latest_path, _stale_name)
                            if os.path.exists(_stale):
                                os.remove(_stale)
                        transformer.save_checkpoint(latest_path, save_full_model=True)

            # ---- Load fine best before Restriction (if more epochs remain) ----
            if global_epoch < total_epoch_budget:
                if not deep_search:
                    if os.path.exists(fine_best_path):
                        logging.info(f"  Loading fine best (acc={fine_best_acc:.3f}%, ep {fine_best_epoch}) for restriction")
                        # Preserve current fine-grid momentum (accumulated over all fine epochs)
                        # because load_from_checkpoint restores momentum from checkpoint save time,
                        # which may be the first fine epoch (near-zero after prolongation).
                        current_momentum = getattr(transformer, 'momentum', None)
                        transformer.load_from_checkpoint(fine_best_path)
                        if current_momentum is not None:
                            transformer.momentum = current_momentum
                else:
                    logging.info(f"  Deep search: continuing from current state (no checkpoint reload)")

            # Reset resume state after first resumed cycle completes
            start_coarse_ep = 0
            start_fine_ep = 0
            resume_phase = None

            # P5: Cross-V-cycle lr adjustment
            if global_epoch < total_epoch_budget:
                cycle_improved = global_best_acc > cycle_start_best

                if exploration_boost:
                    # P5-new: Cyclic Exploration Boost — boost lr on plateau, reset and repeat
                    # Never decay below lr_initial. Rely on V-cycle LoRA re-init for diversity.

                    # Divergence protection: if boost caused >1pp regression, revert
                    boost_reverted = False
                    if boost_count > 0 and not cycle_improved:
                        cycle_best = max(coarse_best_acc, fine_best_acc)
                        if cycle_best < cycle_start_best - 1.0:
                            args.lr = lr_before_boost
                            boost_count = max(0, boost_count - 1)
                            if auto_fine_lr:
                                phase3_lr = args.lr / ratio
                            logging.info(f"  [P5] Divergence detected! cycle_best={cycle_best:.3f}% < start={cycle_start_best:.3f}% - 1pp. "
                                         f"Reverting lr to {args.lr:.2e}, boost_count={boost_count}")
                            no_improve_cycles = 0
                            boost_reverted = True

                    if not boost_reverted:
                        if cycle_improved:
                            no_improve_cycles = 0
                            boost_count = 0
                        else:
                            no_improve_cycles += 1
                            if no_improve_cycles >= lr_decay_patience:
                                lr_before_boost = args.lr
                                old_lr_C = args.lr
                                lr_max = lr_initial * boost_max_ratio
                                if args.lr < lr_max:
                                    # Boost lr to escape plateau
                                    args.lr = min(args.lr * boost_factor, lr_max)
                                    boost_count += 1
                                    if auto_fine_lr:
                                        phase3_lr = args.lr / ratio
                                        logging.info(f"  [P5] Exploration boost #{boost_count}: lr_C {old_lr_C:.2e} -> {args.lr:.2e}, "
                                                     f"lr_F -> {phase3_lr:.2e} (no improvement for {no_improve_cycles} cycles)")
                                    else:
                                        logging.info(f"  [P5] Exploration boost #{boost_count}: lr {old_lr_C:.2e} -> {args.lr:.2e} "
                                                     f"(no improvement for {no_improve_cycles} cycles)")
                                else:
                                    # Already at max — reset to initial lr and start new exploration round
                                    explore_round = getattr(args, '_explore_round', 0) + 1
                                    args._explore_round = explore_round
                                    args.lr = lr_initial
                                    boost_count = 0
                                    if auto_fine_lr:
                                        phase3_lr = args.lr / ratio
                                    logging.info(f"  [P5] Exploration round #{explore_round} complete (at lr cap {lr_max:.2e}). "
                                                 f"Reset lr to {args.lr:.2e}, starting new round")
                                no_improve_cycles = 0

                elif lr_decay_enabled:
                    # P5-legacy: ReduceLROnPlateau
                    if cycle_improved:
                        no_improve_cycles = 0
                    else:
                        no_improve_cycles += 1
                        if no_improve_cycles >= lr_decay_patience and args.lr > lr_decay_min:
                            old_lr_C = args.lr
                            args.lr = max(args.lr * lr_decay_factor, lr_decay_min)
                            lr_decay_count += 1
                            if auto_fine_lr:
                                phase3_lr = args.lr / ratio
                                logging.info(f"  [P5] lr decay #{lr_decay_count}: lr_C {old_lr_C:.2e} -> {args.lr:.2e}, "
                                             f"lr_F -> {phase3_lr:.2e} (no improvement for {no_improve_cycles} cycles)")
                            else:
                                if phase3_lr is not None:
                                    old_lr_F = phase3_lr
                                    phase3_lr = max(phase3_lr * lr_decay_factor, lr_decay_min)
                                    logging.info(f"  [P5] lr decay #{lr_decay_count}: lr_C {old_lr_C:.2e} -> {args.lr:.2e}, "
                                                 f"lr_F {old_lr_F:.2e} -> {phase3_lr:.2e} (no improvement for {no_improve_cycles} cycles)")
                                else:
                                    logging.info(f"  [P5] lr decay #{lr_decay_count}: lr {old_lr_C:.2e} -> {args.lr:.2e} "
                                                 f"(no improvement for {no_improve_cycles} cycles)")
                            no_improve_cycles = 0

            # P6: Val_acc velocity logging (and optional decay)
            # Bug fix: skip V1 for EMA init (V1 gain = 0→X% is artificially high)
            # When exploration_boost is in "exploration" phase, P6 only logs, no decay
            if acc_velocity_decay and global_epoch < total_epoch_budget:
                cycle_epochs = max(1, global_epoch - cycle_start_epoch)
                cycle_gain = global_best_acc - cycle_start_best
                cycle_velocity = cycle_gain / cycle_epochs  # pp per epoch

                if cycle + 1 == 1:
                    # V1: skip EMA init — gain from 0% is meaningless for comparison
                    logging.info(f"  [P6] V1 velocity: {cycle_velocity:.4f}pp/ep "
                                 f"(gain={cycle_gain:.3f}pp in {cycle_epochs}ep, skipped for EMA init)")
                elif not acc_velocity_ema_init:
                    # V2: use as first real EMA data point
                    acc_velocity_ema = cycle_velocity
                    acc_velocity_ema_init = True
                    logging.info(f"  [P6] V{cycle+1} velocity: {cycle_velocity:.4f}pp/ep "
                                 f"(gain={cycle_gain:.3f}pp in {cycle_epochs}ep, EMA init={acc_velocity_ema:.4f})")
                else:
                    prev_ema = acc_velocity_ema
                    acc_velocity_ema = 0.7 * acc_velocity_ema + 0.3 * cycle_velocity

                    # Only decay if exploration_boost is disabled (cyclic exploration never decays)
                    p6_can_decay = not exploration_boost
                    if p6_can_decay and cycle_velocity < 0.3 * prev_ema and cycle_velocity < 0.1 and args.lr > lr_decay_min:
                        old_lr_C = args.lr
                        args.lr = max(args.lr * lr_decay_factor, lr_decay_min)
                        lr_decay_count += 1
                        if auto_fine_lr:
                            phase3_lr = args.lr / ratio
                        elif phase3_lr is not None:
                            phase3_lr = max(phase3_lr * lr_decay_factor, lr_decay_min)
                        logging.info(f"  [P6] Velocity decay #{lr_decay_count}: vel={cycle_velocity:.4f}pp/ep "
                                     f"(EMA={prev_ema:.4f}), lr_C -> {args.lr:.2e}")
                        no_improve_cycles = 0

                    logging.info(f"  [P6] V{cycle+1} velocity: {cycle_velocity:.4f}pp/ep "
                                 f"(gain={cycle_gain:.3f}pp in {cycle_epochs}ep, EMA={acc_velocity_ema:.4f})")

            # ---- Auto Warm Restart (AWR): check after P5/P6 ----
            if auto_warm_restart and global_epoch < total_epoch_budget:
                # Check if this cycle made meaningful improvement (above min_gain threshold)
                cycle_gain_since_restart = global_best_acc - awr_best_at_restart
                if cycle_improved and cycle_gain_since_restart >= warm_restart_min_gain:
                    no_improve_since_restart = 0
                    awr_best_at_restart = global_best_acc
                else:
                    no_improve_since_restart += 1

                if (no_improve_since_restart >= warm_restart_patience
                        and warm_restart_count < max_warm_restarts
                        and os.path.exists(global_best_path)):

                    warm_restart_count += 1
                    old_lr_C = args.lr
                    old_lr_F = phase3_lr if phase3_lr else args.lr

                    # Optionally decay lr
                    if warm_restart_lr_decay and warm_restart_count > warm_restart_lr_decay_after:
                        args.lr = max(args.lr * warm_restart_lr_factor, lr_decay_min)
                        if auto_fine_lr:
                            phase3_lr = args.lr / ratio
                        elif phase3_lr is not None:
                            phase3_lr = max(phase3_lr * warm_restart_lr_factor, lr_decay_min)

                    new_lr_F = phase3_lr if phase3_lr else args.lr
                    logging.info("=" * 60)
                    logging.info(f"[AWR] Auto Warm Restart #{warm_restart_count}/{max_warm_restarts}")
                    logging.info(f"  Trigger: {no_improve_since_restart} V-cycles without global best improvement")
                    logging.info(f"  Loading best checkpoint: {global_best_acc:.3f}% (ep {global_best_epoch})")
                    if args.lr != old_lr_C or new_lr_F != old_lr_F:
                        logging.info(f"  lr decay: lr_C {old_lr_C:.2e} -> {args.lr:.2e}, "
                                     f"lr_F {old_lr_F:.2e} -> {new_lr_F:.2e}")
                    else:
                        logging.info(f"  lr unchanged: lr_C={args.lr:.2e}, lr_F={new_lr_F:.2e}")
                    logging.info("=" * 60)

                    # Load best checkpoint (warm: preserves evec_a + momentum + alpha)
                    transformer.load_from_checkpoint(global_best_path)

                    # Reset P5 counters
                    no_improve_cycles = 0
                    no_improve_since_restart = 0
                    awr_best_at_restart = global_best_acc
                    boost_count = 0

                    # Reset lr_initial for P5 boost ceiling (relative to new lr)
                    if exploration_boost:
                        lr_initial = args.lr

        # ---- V-Cycle Final Summary ----
        logging.info("=" * 60)
        logging.info("V-Cycle Training Complete")
        logging.info(f"  Cycles completed: {cycle + 1}")
        if auto_warm_restart:
            logging.info(f"  Warm restarts: {warm_restart_count}/{max_warm_restarts}")
        logging.info(f"  Total epochs: {global_epoch}")
        logging.info(f"  Best accuracy: {global_best_acc:.3f}% (epoch {global_best_epoch})")
        logging.info(f"  Total forward passes: {total_forward_passes}")
        logging.info("=" * 60)

        if args.logging == "tensorboard" and tb_writer is not None:
            tb_writer.close()
        elif args.logging in ("wandb", "wandb+csv") and tb_writer is not None:
            import wandb
            wandb.finish()
        return

    # ========================================
    # Legacy 3-Phase Training (when V-cycle is disabled)
    # ========================================

    # Resume from checkpoint if specified
    resume_checkpoint = getattr(args, 'resume_checkpoint', None)
    start_epoch = getattr(args, 'resume_epoch', 0)
    skip_phase1 = getattr(args, 'skip_phase1', False)
    resume_phase3 = False  # Flag for Phase 3 resume
    skip_to_phase2 = False  # Flag for Phase 2 skip
    if resume_checkpoint:
        logging.info("=" * 60)
        logging.info(f"Resuming from checkpoint: {resume_checkpoint}")
        transformer.load_from_checkpoint(resume_checkpoint)
        # Check if this is a Phase 3 checkpoint
        if transformer.get_phase() == 'fine':
            resume_phase3 = True
            logging.info(f"  Phase 3 checkpoint detected, skipping Phase 1 and 2")
            logging.info(f"  Starting Phase 3 from epoch {start_epoch + 1}")
        elif skip_phase1:
            skip_to_phase2 = True
            logging.info(f"  --skip_phase1: Will skip Phase 1, go directly to Phase 2 + 3")
            logging.info(f"  Checkpoint loaded as Phase 1 best result")
        else:
            logging.info(f"  Starting Phase 1 from epoch {start_epoch + 1}")
        logging.info("=" * 60)

    # Skip Phase 1 and 2 if resuming from Phase 3
    if not resume_phase3 and not skip_to_phase2:
        # ========================================
        # Phase 1: Coarse Grid Training
        # ========================================
        logging.info("=" * 60)
        logging.info("Phase 1: Coarse Grid LoRA Training")
        logging.info(f"  Epochs: {start_epoch + 1} to {args.epochs}")
        logging.info(f"  LoRA rank: {transformer.lora_rank}")
        logging.info(f"  VMS feedback: {'enabled' if transformer.use_fine_residual else 'disabled'}")
        logging.info(f"  Phase 1 patience: {phase1_patience}")
        logging.info(f"  Checkpoint interval: every {checkpoint_interval} epochs")
        logging.info("=" * 60)

        transformer.set_phase('coarse')

        # Save initial model weights at Phase 1 start (for checkpoint recovery)
        # This ensures best_checkpoint can always be restored even if periodic checkpoint not saved
        initial_model_path = os.path.join(checkpoint_dir, "initial_model.pt")
        if not os.path.exists(initial_model_path):
            torch.save(transformer.model.state_dict(), initial_model_path)
            logging.info(f"  Initial model saved to {initial_model_path}")

        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            log_dict['epoch'] = epoch
            log_dict['phase'] = 1

            # Set current epoch for τ warmup
            transformer.set_epoch(epoch + 1)

            # Validation
            if (epoch + 1) % args.eval_every == 0:
                reset_log_dict(log_dict)
                val_loss_mean, val_acc = evaluate(transformer, val_dataloader, device, log_dict)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    patience_counter = 0

                    # Save best checkpoint (keep only 1 latest best)
                    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_phase1_epoch{epoch+1}")
                    transformer.save_checkpoint(best_checkpoint_path)
                    logging.info(f"  → New best! Saved checkpoint to {best_checkpoint_path}")

                    # Delete previous best checkpoint to save disk space
                    if prev_best_phase1_path and os.path.exists(prev_best_phase1_path):
                        shutil.rmtree(prev_best_phase1_path)
                        logging.info(f"  → Deleted previous best checkpoint: {prev_best_phase1_path}")
                    prev_best_phase1_path = best_checkpoint_path
                else:
                    patience_counter += 1

                log_dict["best_val_acc"] = best_val_acc
                if tb_writer:
                    tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

                logging.info(f"[Phase 1][Epoch {epoch + 1}] val_loss: {val_loss_mean:.5f}, "
                            f"val_acc: {val_acc:.3f}%, best: {best_val_acc:.3f}% (epoch {best_epoch})")

                # Check for Phase 1 → Phase 2 transition
                if patience_counter >= phase1_patience:
                    logging.info(f"Phase 1 complete: No improvement for {phase1_patience} epochs")
                    logging.info(f"Best Phase 1 accuracy: {best_val_acc:.3f}% at epoch {best_epoch}")
                    break

                # Early stopping (optional)
                if args.early_stopping and patience_counter >= args.patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}.")
                    break

            # Training loop
            transformer.model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_dataloader):
                reset_log_dict(log_dict)
                b = {k: v.to(device) for k, v in batch.items()}
                transformer.training_step(model, b, tb_writer=tb_writer, **kwargs)
                total_forward_passes += transformer.state.query[-1]
                log_dict['num_forward_passes'] = total_forward_passes
                train_loss += log_dict.get('train_loss', 0.0)

                if tb_writer:
                    tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

            alloc_mem = transformer.measure_memory_usage()
            epoch_time = time.time() - epoch_start_time
            log_dict['memory_gb'] = alloc_mem
            log_dict['epoch_time_sec'] = epoch_time
            if tb_writer:
                tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

            train_loss /= len(train_dataloader)
            # VMS diagnostic summary per epoch
            vms_info = ''
            if transformer.use_fine_residual and 'vms_c_B_mean' in log_dict:
                vms_info = (f", VMS: c_B={log_dict.get('vms_c_B_mean', 0):.4e}"
                           f" c_A={log_dict.get('vms_c_A_mean', 0):.4e}"
                           f" c_f={log_dict.get('vms_c_f_raw_mean', 0):.4e}"
                           f" c_f'={log_dict.get('vms_c_f_prime_raw_mean', 0):.4e}"
                           f" τ={log_dict.get('vms_tau_mean', 0):.3f}"
                           f" adj_B={log_dict.get('vms_adj_ratio_B_mean', 0):.3f}"
                           f" adj_A={log_dict.get('vms_adj_ratio_A_mean', 0):.3f}"
                           f" clip={log_dict.get('vms_clipped_count', 0)}")
            ct_info = ''
            if getattr(transformer, 'curvature_transfer', False) and 'ct_rho_mean' in log_dict:
                ct_info = (f", CT: ρ={log_dict.get('ct_rho_mean', 0):.3f}"
                          f" nA={log_dict.get('ct_norm_A_mean', 0):.3e}"
                          f" nB={log_dict.get('ct_norm_B_mean', 0):.3e}")
            lr_cov_info = ''
            if 'lr_cov_now' in log_dict:
                lr_cov_info = f", lr_cov={log_dict['lr_cov_now']:.2e}"
            logging.info(f'[Phase 1][Epoch {epoch + 1}] Avg. train loss: {train_loss:.5f}, mem: {alloc_mem:.2f}GB{vms_info}{ct_info}{lr_cov_info}')

            # Periodic checkpoint saving
            # First checkpoint: save full model + LoRA + 1D params (~262 MB)
            # Subsequent checkpoints: save LoRA + 1D params only (~11 MB)
            if (epoch + 1) % checkpoint_interval == 0:
                periodic_checkpoint_path = os.path.join(checkpoint_dir, f"periodic_epoch{epoch+1}")
                save_full = not first_periodic_saved
                transformer.save_checkpoint(periodic_checkpoint_path, save_full_model=save_full)
                if save_full:
                    first_periodic_saved = True
                    logging.info(f"  → First periodic checkpoint (full model) saved to {periodic_checkpoint_path}")
                else:
                    logging.info(f"  → Periodic checkpoint (LoRA+1D) saved to {periodic_checkpoint_path}")

            # Periodic LoRA absorption (multi V-cycle)
            # Theory: claude-vms-theory-2026-03-02-markdown.md §10.4
            # Absorb LoRA into W_0 to keep fine-grid and coarse-grid synchronized
            absorption_interval = getattr(args, 'absorption_interval', 0)
            if absorption_interval > 0 and (epoch + 1) % absorption_interval == 0 and (epoch + 1) < args.epochs:
                logging.info(f"  ★ V-cycle absorption at epoch {epoch + 1}")
                transformer.absorb_lora()

        phase1_final_acc = best_val_acc
        phase1_queries = total_forward_passes

        # ========================================
        # Phase 2: Fine Grid Reconstruction
        # ========================================
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            logging.info("=" * 60)
            logging.info("Phase 2: Fine Grid Reconstruction")
            logging.info(f"  Loading best checkpoint from epoch {best_epoch}")
            logging.info("=" * 60)

            transformer.set_phase('reconstruction')

            # Load best checkpoint
            transformer.load_from_checkpoint(best_checkpoint_path)

            # Get a batch for reconstruction
            reconstruction_batch = next(iter(train_dataloader))
            reconstruction_batch = {k: v.to(device) for k, v in reconstruction_batch.items()}

            # Run reconstruction
            transformer.reconstruct_fine_grid(model, reconstruction_batch, num_smoothing_steps=reconstruction_steps)

            # Validate after reconstruction
            reset_log_dict(log_dict)
            val_loss_mean, val_acc = evaluate(transformer, val_dataloader, device, log_dict)
            logging.info(f"[Phase 2] After reconstruction: val_loss: {val_loss_mean:.5f}, val_acc: {val_acc:.3f}%")

            # Set initial Phase 3 accuracy from reconstruction
            phase3_best_acc = val_acc
        else:
            logging.info("No checkpoint available for Phase 2/3, using Phase 1 results only")
            phase1_final_acc = best_val_acc
            phase3_best_acc = 0.0

    # ========================================
    # Phase 2 from --skip_phase1: checkpoint already loaded
    # ========================================
    if skip_to_phase2:
        logging.info("=" * 60)
        logging.info("Phase 2: Fine Grid Reconstruction (--skip_phase1)")
        logging.info(f"  Checkpoint: {resume_checkpoint}")
        logging.info("=" * 60)

        # Validate before reconstruction to confirm checkpoint accuracy
        reset_log_dict(log_dict)
        val_loss_mean, val_acc = evaluate(transformer, val_dataloader, device, log_dict)
        logging.info(f"[Phase 2] Before reconstruction: val_loss: {val_loss_mean:.5f}, val_acc: {val_acc:.3f}%")
        phase1_final_acc = val_acc

        transformer.set_phase('reconstruction')

        # Get a batch for reconstruction
        reconstruction_batch = next(iter(train_dataloader))
        reconstruction_batch = {k: v.to(device) for k, v in reconstruction_batch.items()}

        # Run reconstruction
        transformer.reconstruct_fine_grid(model, reconstruction_batch, num_smoothing_steps=reconstruction_steps)

        # Validate after reconstruction
        reset_log_dict(log_dict)
        val_loss_mean, val_acc = evaluate(transformer, val_dataloader, device, log_dict)
        logging.info(f"[Phase 2] After reconstruction: val_loss: {val_loss_mean:.5f}, val_acc: {val_acc:.3f}%")
        phase3_best_acc = val_acc

    # ========================================
    # Phase 3: Fine Grid Training
    # ========================================
    # Run Phase 3 if: (1) coming from Phase 2, (2) resuming from Phase 3, or (3) --skip_phase1
    run_phase3 = resume_phase3 or skip_to_phase2 or (not resume_phase3 and not skip_to_phase2 and best_checkpoint_path and os.path.exists(best_checkpoint_path))

    if run_phase3:
        if resume_phase3:
            # Resuming from Phase 3 checkpoint
            phase3_start_epoch = start_epoch
            # Validate current state
            reset_log_dict(log_dict)
            val_loss_mean, val_acc = evaluate(transformer, val_dataloader, device, log_dict)
            phase3_best_acc = val_acc
            phase1_final_acc = 0.0  # Unknown from resume
            logging.info("=" * 60)
            logging.info("Phase 3: Fine Grid LOREN Training (Resumed)")
            logging.info(f"  Starting from epoch: {phase3_start_epoch + 1}")
            logging.info(f"  Current val_acc: {val_acc:.3f}%")
            logging.info(f"  Remaining epochs: {phase3_epochs - phase3_start_epoch}")
            logging.info(f"  Learning rate: {phase3_lr if phase3_lr else args.lr}")
            logging.info(f"  Checkpoint interval: every {phase3_checkpoint_interval} epochs (full model)")
            logging.info("=" * 60)
        else:
            # Coming from Phase 2
            phase3_start_epoch = 0
            logging.info("=" * 60)
            logging.info("Phase 3: Fine Grid LOREN Training")
            logging.info(f"  Epochs: {phase3_epochs}")
            logging.info(f"  Learning rate: {phase3_lr if phase3_lr else args.lr}")
            logging.info(f"  Checkpoint interval: every {phase3_checkpoint_interval} epochs (full model)")
            logging.info("=" * 60)

        transformer.set_phase('fine')
        phase3_patience_counter = 0
        phase3_best_epoch = phase3_start_epoch  # Track best epoch
        prev_best_phase3_path = None  # Track previous best checkpoint for cleanup

        # Override learning rate for Phase 3 (full-parameter LOREN needs smaller lr)
        if phase3_lr is not None:
            phase1_lr = transformer.state.learning_rate
            transformer.state.learning_rate = phase3_lr
            logging.info(f"  Phase 3 lr: {phase3_lr} (Phase 1 was {phase1_lr})")

        for epoch in range(phase3_start_epoch, phase3_epochs):
            epoch_start_time = time.time()
            log_dict['epoch'] = epoch
            log_dict['phase'] = 3

            # Set current epoch for τ warmup (Phase 3 continues from Phase 1 epoch count)
            transformer.set_epoch(epoch + 1)

            # Validation
            if (epoch + 1) % args.eval_every == 0:
                reset_log_dict(log_dict)
                val_loss_mean, val_acc = evaluate(transformer, val_dataloader, device, log_dict)

                if val_acc > phase3_best_acc:
                    phase3_best_acc = val_acc
                    phase3_best_epoch = epoch + 1
                    phase3_patience_counter = 0

                    # Save best Phase 3 checkpoint (keep only 1 latest best)
                    best_phase3_path = os.path.join(checkpoint_dir, f"best_phase3_epoch{epoch+1}")
                    transformer.save_checkpoint(best_phase3_path)
                    logging.info(f"  → New best! Saved checkpoint to {best_phase3_path}")

                    # Delete previous best checkpoint to save disk space
                    if prev_best_phase3_path and os.path.exists(prev_best_phase3_path):
                        shutil.rmtree(prev_best_phase3_path)
                        logging.info(f"  → Deleted previous best checkpoint: {prev_best_phase3_path}")
                    prev_best_phase3_path = best_phase3_path
                else:
                    phase3_patience_counter += 1

                log_dict["best_val_acc"] = phase3_best_acc
                if tb_writer:
                    tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

                logging.info(f"[Phase 3][Epoch {epoch + 1}] val_loss: {val_loss_mean:.5f}, "
                            f"val_acc: {val_acc:.3f}%, best: {phase3_best_acc:.3f}%")

                # Early stopping for Phase 3
                if args.early_stopping and phase3_patience_counter >= args.patience:
                    logging.info(f"Phase 3 early stopping at epoch {epoch + 1}")
                    break

            # Fine grid training (use parent LOREN training step)
            transformer.model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_dataloader):
                reset_log_dict(log_dict)
                b = {k: v.to(device) for k, v in batch.items()}
                # Use parent class training_step for fine grid
                from model.glue_transformer_loren import GLUETransformerLOREN
                GLUETransformerLOREN.training_step(transformer, model, b, tb_writer=tb_writer, **kwargs)
                total_forward_passes += transformer.state.query[-1]
                log_dict['num_forward_passes'] = total_forward_passes
                train_loss += log_dict.get('train_loss', 0.0)
                if tb_writer:
                    tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

            alloc_mem = transformer.measure_memory_usage()
            epoch_time = time.time() - epoch_start_time
            log_dict['memory_gb'] = alloc_mem
            log_dict['epoch_time_sec'] = epoch_time
            if tb_writer:
                tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

            train_loss /= len(train_dataloader)
            logging.info(f'[Phase 3][Epoch {epoch + 1}] Avg. train loss: {train_loss:.5f}, mem: {alloc_mem:.2f}GB')

            # Periodic checkpoint saving for Phase 3 (full model, less frequent)
            if (epoch + 1) % phase3_checkpoint_interval == 0:
                periodic_checkpoint_path = os.path.join(checkpoint_dir, f"phase3_epoch{epoch+1}")
                transformer.save_checkpoint(periodic_checkpoint_path)
                logging.info(f"  → Periodic checkpoint (full model) saved to {periodic_checkpoint_path}")

        best_val_acc = max(phase1_final_acc, phase3_best_acc)

    # ========================================
    # Final Summary
    # ========================================
    logging.info("=" * 60)
    logging.info("EV-VMS Training Complete")
    if not resume_phase3:
        logging.info(f"  Phase 1 best accuracy: {phase1_final_acc:.3f}% (epoch {best_epoch})")
    if run_phase3:
        logging.info(f"  Phase 3 best accuracy: {phase3_best_acc:.3f}% (epoch {phase3_best_epoch})")
    logging.info(f"  Overall best accuracy: {best_val_acc:.3f}%")
    logging.info(f"  Total forward passes: {total_forward_passes}")
    logging.info("=" * 60)

    if args.logging == "tensorboard" and tb_writer is not None:
        tb_writer.close()
    elif args.logging in ("wandb", "wandb+csv") and tb_writer is not None:
        import wandb
        wandb.finish()
