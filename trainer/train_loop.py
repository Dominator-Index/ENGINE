import time

import numpy as np
import torch
import lightning as L
from lightning.pytorch import Trainer

from common.logging import logger as logging


def reset_log_dict(log_dict):
    """Clear metric keys between logging points.

    Keys are removed (not zeroed) so that only freshly-written metrics
    appear in the next log_metrics() call, preventing spurious zeros
    in wandb/CSV (e.g. train_loss=0 at validation steps).
    """
    for k in ('train_loss', 'val_loss', 'val_acc', 'time'):
        log_dict.pop(k, None)
    for k in list(log_dict):
        if k.startswith(('subzero_', 'agzo_', 'zogv_', 'zomuon_')):
            log_dict.pop(k, None)

def finetune_FO(args,
                device: torch.device,
                dm: L.LightningDataModule,
                transformer: L.LightningModule, tb_writer=None):
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    log_dict = {}

    if args.bf16:
        transformer.to(torch.bfloat16)

    model = transformer.model.to(device)

    transformer.configure_params()
    optimizer = transformer.configure_optimizers()

    transformer.model.eval()
    if args.early_stopping:
        best_val_acc = 0
        patience_counter = 0

    global_step = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        log_dict['epoch'] = epoch

        if (epoch + 1) % args.eval_every == 0:
            # validation loop
            reset_log_dict(log_dict)
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

            if args.early_stopping:
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                log_dict["best_val_acc"] = best_val_acc

            tb_writer.log_metrics(log_dict, step=global_step)

            if args.early_stopping and patience_counter >= args.patience:
                logging.info(
                    f"Early stopping at epoch {epoch+1}. Best validation accuracy: {best_val_acc}"
                )
                break

        # training loop
        transformer.model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            reset_log_dict(log_dict)
            b = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**b)
            loss = outputs[0]
            loss.backward()
            alloc_mem = transformer.measure_memory_usage()
            optimizer.step()

            train_loss += loss.item()
            global_step += 1

            log_dict['train_loss'] = loss.item()
            log_dict['time'] = 0.0  # or measure if you like

            if tb_writer:
                tb_writer.log_metrics(log_dict, step=global_step)
        epoch_time = time.time() - epoch_start_time
        log_dict['memory_gb'] = alloc_mem
        log_dict['epoch_time_sec'] = epoch_time
        log_dict['num_forward_passes'] = transformer.state.num_forward_passes
        if tb_writer:
            tb_writer.log_metrics(log_dict, step=global_step)

        logging.info(f"[{epoch + 1:}] val_loss: {val_loss_mean:.5f}, val_acc: {val_acc:.3f}, "
                     f"mem: {alloc_mem:.5f}")
        avg_train_loss = train_loss / len(train_dataloader)
        logging.info(f'Avg. train loss: {avg_train_loss:.5f}')

    if args.logging == "tensorboard" and tb_writer is not None:
        tb_writer.close()
    elif args.logging in ("wandb", "wandb+csv") and tb_writer is not None:
        import wandb
        wandb.finish()

def finetune_ZO(args,
                device: torch.device,
                dm: L.LightningDataModule,
                transformer: L.LightningModule, tb_writer=None):
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    log_dict = {}
    kwargs = {'log': log_dict}

    if args.bf16:
        transformer.to(torch.bfloat16)

    model = transformer.model.to(device)

    # Algorithm-specific routines
    if args.algorithm == 'ZOSVRG':
        total_batches = len(train_dataloader)
        fb_train_loader = dm.train_full_dataloader()
        kwargs['full_batch_size'] = total_batches
    elif args.algorithm == 'ZOAdam':
        if args.low_bit_adam == 8:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(model.parameters(),
                                           lr=args.lr,
                                           eps=args.eps,
                                           weight_decay=0)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
        kwargs['optimizer'] = optimizer

    transformer.configure_params()

    transformer.model.eval()
    if args.early_stopping:
        best_val_acc = 0
        patience_counter = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        log_dict['epoch'] = epoch

        if (epoch + 1) % args.eval_every == 0:
            # validation loop
            reset_log_dict(log_dict)
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

            if args.early_stopping:
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                log_dict["best_val_acc"] = best_val_acc

            tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

            if args.early_stopping and patience_counter >= args.patience:
                logging.info(
                    f"Early stopping at epoch {epoch+1}. Best validation accuracy: {best_val_acc}"
                )
                break

        # training loop
        transformer.model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            reset_log_dict(log_dict)

            if args.algorithm == 'ZOSVRG':
                curr_iter = epoch * total_batches + batch_idx
                kwargs['curr_iter'] = curr_iter

                if (curr_iter % args.q) == 0:
                    batch = next(iter(fb_train_loader))
                    kwargs['is_full_batch'] = True
                else:
                    kwargs['is_full_batch'] = False

            b = {k: v.to(device) for k, v in batch.items()}
            transformer.training_step(model, b, tb_writer=tb_writer, **kwargs)

            if kwargs.get('is_full_batch', False):
                train_loss += log_dict.get('fb_train_loss', 0.0)
            else:
                train_loss += log_dict.get('train_loss', 0.0)

            if tb_writer:
                tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

        alloc_mem = transformer.measure_memory_usage()
        epoch_time = time.time() - epoch_start_time
        log_dict['memory_gb'] = alloc_mem
        log_dict['epoch_time_sec'] = epoch_time
        log_dict['num_forward_passes'] = transformer.state.num_forward_passes
        if tb_writer:
            tb_writer.log_metrics(log_dict, step=transformer.state.global_training_steps)

        logging.info(f"[{epoch + 1:}] val_loss: {val_loss_mean:.5f}, val_acc: {val_acc:.3f}, "
                    f"mem: {alloc_mem:.5f}")
        train_loss /= len(train_dataloader)
        logging.info(f'Avg. train loss: {train_loss:.5f}')

    if args.logging == "tensorboard" and tb_writer is not None:
        tb_writer.close()
    elif args.logging in ("wandb", "wandb+csv") and tb_writer is not None:
        import wandb
        wandb.finish()
