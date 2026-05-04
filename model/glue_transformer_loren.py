import math
import time
from typing import Optional

import numpy as np
import torch

from .glue_transformer_lozo import GLUETransformerLoZo
from common.alg_utils import sgd_with_wdecay
from common.logging import logger as evec_logger


class GLUETransformerLOREN(GLUETransformerLoZo):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        zero_order_eps: float = 1e-3,
        learning_rate_aux: float = 1e-3,
        minibatch: int = 64,
        max_norm: float = 18000.0,
        z_std: float = 1.0,
        lr_anneal: float = 1.0,
        full_parameter: bool = True,
        batchsize_limit: int = 16,
        eval_splits: Optional[list] = None,
        soft_prompt: bool = False,
        use_SGD: bool = True,
        logger_type: str = 'wandb',
        hf_token: Optional[str] = None,
        lora: bool = False,
        model_init_seed: Optional[int] = None,
        weight_decay: float = 0.01,
        damping: float = 0.1,
        lr_cov: float = 2e-3,
        beta1: float = 0.9,
        num_queries: int = 10,
        **kwargs,
    ):
        super().__init__(model_name_or_path,
                         num_labels,
                         zero_order_eps,
                         learning_rate_aux,
                         minibatch,
                         max_norm,
                         z_std,
                         lr_anneal,
                         full_parameter,
                         batchsize_limit,
                         eval_splits,
                         soft_prompt,
                         use_SGD,
                         logger_type,
                         hf_token,
                         lora,
                         model_init_seed,
                         weight_decay)

        self.damping = damping
        self.lr_cov = lr_cov
        self.d_sqrt = math.sqrt(damping)
        self.evec_a = {}
        self.momentum = {}
        self.num_queries = num_queries
        self.beta1 = beta1
        self.param_info = {}

        # evec_a diagnostic
        self._evec_a_diagnostic = kwargs.get('evec_a_diagnostic', False)
        self._evec_a_diag_interval = kwargs.get('evec_a_diag_interval', 4)
        self._evec_a_prev = {}

        assert num_queries >= 2, 'At least 2 samples are required for RLOO estimation'

    def configure_params(self):
        super().configure_params()
        for name, param in self.params:
            param_shape = param.shape
            param_flat_size = param.numel()
            evec_size = param.size(1) if param.ndim >= 2 else param_flat_size
            a = torch.randn(evec_size, device=param.device, dtype=param.dtype)
            self.evec_a[name] = a
            if param.ndim >= 2:
                self.momentum[name] = torch.zeros_like(param)
            self.param_info[name] = {
                'shape': param_shape,
                'flat_size': param_flat_size,
                'evec_size': evec_size,
                'mfactor': param_flat_size // evec_size
            }

    @torch.no_grad()
    def perturb_parameters(self, random_seed: int, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.params:
            info = self.param_info[name]
            shape = info['shape']
            a = self.evec_a[name]
            a_flat = a.view(1, -1).expand(info['mfactor'], -1).reshape(-1)
            u_vec = torch.randn_like(param).view(-1)
            a_flat = a_flat.to(dtype=u_vec.dtype)
            sq_norm_a = torch.dot(a_flat, a_flat)
            dot_au = torch.dot(a_flat, u_vec)
            damping_sq_norm_a = math.sqrt(self.damping + sq_norm_a.item())
            alpha = (self.d_sqrt + damping_sq_norm_a) * dot_au / (sq_norm_a * damping_sq_norm_a)
            sqrt_cov_u = u_vec.sub(alpha * a_flat).view(shape)
            param.add_(sqrt_cov_u, alpha=scaling_factor * self.zero_order_eps)

    @torch.no_grad()
    def lnes_zo_update(self, random_seeds, fvals, fmean):
        device = next(self.model.parameters()).device
        fweight = torch.tensor(fvals, device=device) - fmean
        num_queries_inv = 1.0 / (self.num_queries - 1)
        eps_inv = 1.0 / (self.zero_order_eps * (self.num_queries - 1))

        for k in range(self.num_queries):
            torch.manual_seed(random_seeds[k])
            for name, param in self.params:
                info = self.param_info[name]
                shape = info['shape']
                a = self.evec_a[name]
                a_flat = a.view(1, -1).expand(info['mfactor'], -1).reshape(-1)
                u_vec = torch.randn_like(param).view(-1)
                a_flat = a_flat.to(dtype=u_vec.dtype)
                sq_norm_a = torch.dot(a_flat, a_flat)
                dot_au = torch.dot(a_flat, u_vec)
                damping_sq_norm_a = math.sqrt(self.damping + sq_norm_a.item())

                alpha = (self.d_sqrt + damping_sq_norm_a) * dot_au / (sq_norm_a * damping_sq_norm_a)
                sqrt_cov_u = u_vec.sub(alpha * a_flat).view(shape)

                gx = sqrt_cov_u.mul(fweight[k] * eps_inv)
                if param.ndim >= 2:
                    self.momentum[name].mul_(self.beta1).add_(gx)
                    gx = self.momentum[name]
                sgd_with_wdecay(param, gx, self.state.learning_rate, self.weight_decay)

                c1 = (dot_au * u_vec - a_flat)
                c2 = (self.d_sqrt + damping_sq_norm_a) * (dot_au ** 2 - sq_norm_a) * a_flat / (sq_norm_a * damping_sq_norm_a)
                ak_grad = fweight[k] * (self.zero_order_eps ** 2) * (c1 - c2) / damping_sq_norm_a
                ak_grad = ak_grad.view(shape).sum(0) * num_queries_inv
                sgd_with_wdecay(self.evec_a[name], ak_grad.to(dtype=a.dtype), self.lr_cov, self.weight_decay)

        # evec_a diagnostic
        if (self._evec_a_diagnostic and
                self.state.global_training_steps % self._evec_a_diag_interval == 0):
            self._log_evec_a_diagnostic()

    def _log_evec_a_diagnostic(self):
        """Log evec_a evolution metrics: norm, step_cosine, update_ratio."""
        norms = []
        step_cosines = []
        update_ratios = []

        for name in self.evec_a:
            a = self.evec_a[name]
            a_norm = a.norm().item()
            norms.append(a_norm)

            if name in self._evec_a_prev:
                a_prev = self._evec_a_prev[name]
                a_prev_norm = a_prev.norm()
                if a_norm > 1e-10 and a_prev_norm > 1e-10:
                    cos = torch.dot(a, a_prev) / (a_norm * a_prev_norm)
                    step_cosines.append(cos.abs().item())
                ur = (a - a_prev).norm().item() / (a_norm + 1e-10)
                update_ratios.append(ur)

            self._evec_a_prev[name] = a.clone()

        step = self.state.global_training_steps
        avg_norm = sum(norms) / len(norms) if norms else 0
        cos_str = f"{sum(step_cosines)/len(step_cosines):.10f}" if step_cosines else "N/A"
        ur_str = f"{sum(update_ratios)/len(update_ratios):.8f}" if update_ratios else "N/A"
        gap_str = f"{1.0 - sum(step_cosines)/len(step_cosines):.2e}" if step_cosines else "N/A"
        evec_logger.info(f"    [evec_a] step={step}: avg_norm={avg_norm:.4f}, "
                        f"avg_step_cos={cos_str}, 1-cos={gap_str}, avg_ur={ur_str}")

        if step_cosines and len(step_cosines) >= 3:
            min_cos = min(step_cosines)
            max_cos = max(step_cosines)
            evec_logger.info(f"      step_cos range: [{min_cos:.10f}, {max_cos:.10f}], "
                            f"n_layers={len(step_cosines)}")

    def training_step(self, model, batch, tb_writer=None, **kwargs):
        n = batch['input_ids'].size(0)
        start_time = time.perf_counter()
        random_seeds = [np.random.randint(1000000000) for _ in range(self.num_queries)]
        log_dict = kwargs['log']

        fvals = []
        with torch.inference_mode():
            for k in range(self.num_queries):
                self.perturb_parameters(random_seeds[k], scaling_factor=1)
                loss = self.zo_forward_memory_eff(model, batch)
                fvals.append(loss.item())
                self.perturb_parameters(random_seeds[k], scaling_factor=-1)

        fmean = np.mean(fvals)
        self.lnes_zo_update(random_seeds, fvals, fmean)
        self.state.global_training_steps += 1

        # Store fvals for subclass terrain sensing
        self._last_fvals = fvals
        self._last_fmean = fmean

        total_time = time.perf_counter() - start_time

        self.state.time.append(total_time)
        # Query count = number of forward passes (NOT multiplied by batch size)
        self.state.query.append(self.num_queries)

        log_dict['train_loss'] = fmean.item()
        log_dict['time'] = total_time
