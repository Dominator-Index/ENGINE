"""
AdaSmooth-ZO Implementation - Ultimate Memory Optimization

Integrates all memory optimizations:
1. No candidate storage (saves 12+ GB)
2. fp16 for L matrices (saves 50% L memory)
3. Zero memory leak (detach + explicit del)
4. Aggressive cleanup (gc + cache clearing)
5. CPU weight computation
6. Limited history lists

Total memory for GPT-2-XL:
- K=2: ~15 GB (down from 40 GB!)
- K=6: ~16-18 GB (usable on A6000 47GB)
"""

import math
import time
from typing import Optional
import gc

import numpy as np
import torch

from .glue_transformer_lozo import GLUETransformerLoZo


class GLUETransformerAdaSmooth(GLUETransformerLoZo):
    """
    AdaSmooth-ZO: Ultimate memory-optimized implementation

    Key optimizations:
    - No candidate storage
    - fp16 L matrices
    - Zero memory leak
    - Aggressive cleanup
    """

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
        num_queries: int = 6,
        rank: int = None,
        beta_init: float = 1.0,
        beta_decay: float = 0.05,
        task_name: Optional[str] = None,
        num_epochs: int = 1,
        use_fp16_L: bool = True,  # Use fp16 for L matrices by default
        **kwargs,
    ):
        """
        Initialize AdaSmooth-ZO optimizer.

        Args:
            rank: Low-rank dimension K. Must equal num_queries.
            beta_init: Initial temperature β_0
            beta_decay: Temperature decay rate
            use_fp16_L: Use fp16 for L matrices (saves 50% memory)
        """
        if rank is not None and rank != num_queries:
            raise ValueError(
                f"AdaSmooth requires rank ({rank}) to equal num_queries ({num_queries})"
            )

        super().__init__(
            model_name_or_path,
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
            step_interval=50,
            rank=rank or num_queries,
            weight_decay=weight_decay
        )

        self.K = rank if rank is not None else num_queries
        self.num_queries = num_queries
        self.beta_init = beta_init
        self.beta_decay = beta_decay
        self.use_fp16_L = use_fp16_L

        self.L_matrices = {}
        self.param_info = {}

    def configure_params(self):
        """
        Initialize L matrices with optional fp16 storage.
        """
        super().configure_params()

        total_L_memory = 0
        for name, param in self.params:
            param_shape = param.shape
            d = param.numel()

            # Use fp16 for L matrices to save 50% memory
            L_dtype = torch.float16 if self.use_fp16_L else param.dtype

            self.L_matrices[name] = torch.randn(
                d, self.K,
                device=param.device,
                dtype=L_dtype
            ) * 0.01

            self.param_info[name] = {
                'shape': param_shape,
                'd': d
            }

            # Track memory usage
            L_mem = d * self.K * (2 if self.use_fp16_L else 4) / 1024**3
            total_L_memory += L_mem

        print(f"[AdaSmooth] L matrices: {total_L_memory:.2f} GB ({self.use_fp16_L and 'fp16' or 'fp32'})")

    @torch.no_grad()
    def adasmooth_step(self, model, batch):
        """
        Zero memory leak iteration with fp16 support.

        Memory optimization:
        1. No candidate storage
        2. fp16 L matrices
        3. Explicit detach() + del
        4. Python float instead of tensor
        5. Aggressive cleanup
        """
        # Save original parameters (with detach to break computation graph)
        theta_old = {}
        for name, param in self.params:
            theta_old[name] = param.data.detach().clone()

        # Storage (tiny memory footprint)
        latent_directions = []
        losses = []  # Will be Python floats

        # === Phase 1: Sampling and Evaluation ===
        for k in range(self.K):
            # Sample u_k ~ N(0, I_K)
            u_k = torch.randn(self.K, device=model.device, dtype=torch.float32)
            latent_directions.append(u_k.detach().clone())  # Detach to break graph

            # Apply perturbations: x_k = θ_t + L_t @ u_k
            for name, param in self.params:
                L = self.L_matrices[name]

                # Convert fp16 L to fp32 for computation
                L_compute = L.float() if self.use_fp16_L else L
                delta = torch.matmul(L_compute, u_k)

                # Update parameter to x_k
                param.data.copy_((theta_old[name].view(-1) + delta).view(param.shape))

                # Critical: delete delta immediately
                del delta

            # Delete u_k after use
            del u_k

            # Evaluate loss at x_k
            with torch.inference_mode():
                loss = self.zo_forward_memory_eff(model, batch)
                losses.append(float(loss.item()))  # Convert to Python float!
                del loss  # Delete tensor

            # Aggressive periodic cleanup
            if k > 0 and k % 2 == 0:
                torch.cuda.empty_cache()

        # Restore parameters to θ_t
        for name, param in self.params:
            param.data.copy_(theta_old[name])

        # === Phase 2: Compute Weights (CPU computation) ===
        beta_t = self.beta_init / (1 + self.beta_decay * self.state.global_training_steps)

        # Use Python floats only (no tensors)
        y_list = losses  # Already Python floats
        v_list = [math.exp(-y / beta_t) for y in y_list]
        v_sum = sum(v_list)
        w_list = [v / v_sum for v in v_list]

        # Convert to GPU tensor only when needed
        w_gpu = torch.tensor(w_list, device=model.device, dtype=torch.float32)

        # === Phase 3: Update Mean θ_{t+1} ===
        for name, param in self.params:
            L = self.L_matrices[name]
            L_compute = L.float() if self.use_fp16_L else L

            # Compute weighted sum of latent directions
            weighted_u = torch.zeros(self.K, device=model.device, dtype=torch.float32)
            for k in range(self.K):
                weighted_u.add_(latent_directions[k], alpha=w_list[k])

            # Update: θ_{t+1} = θ_t + L @ weighted_u
            delta = torch.matmul(L_compute, weighted_u)
            param.data.copy_((theta_old[name].view(-1) + delta).view(param.shape))

            # Cleanup
            del delta, weighted_u

        # === Phase 4: Update L Matrix ===
        for name, param in self.params:
            theta_new = param.data.view(-1).detach().clone()
            L_old = self.L_matrices[name]
            L_old_compute = L_old.float() if self.use_fp16_L else L_old

            # Allocate new L
            L_new = torch.empty_like(L_old)

            # Residual base: θ_t - θ_{t+1}
            residual_base = theta_old[name].view(-1) - theta_new

            # Update each column
            for k in range(self.K):
                # Reconstruct x_k = θ_t + L_old @ u_k
                delta_k = torch.matmul(L_old_compute, latent_directions[k])

                # Residual: x_k - θ_{t+1}
                residual = residual_base + delta_k

                # Weighted residual: c_k = √w_k * residual
                # Convert to fp16 if needed
                weighted_residual = torch.sqrt(w_gpu[k]) * residual
                L_new[:, k].copy_(weighted_residual.to(L_new.dtype))

                # Cleanup
                del delta_k, residual, weighted_residual

            # Replace L matrix (old L will be garbage collected)
            self.L_matrices[name] = L_new

            # Cleanup
            del theta_new, L_old, L_old_compute, residual_base, L_new

        # === Final Cleanup ===
        del theta_old
        del latent_directions
        del w_gpu

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        # Return Python objects (no tensors!)
        return losses, w_list, None

    def training_step(self, model, batch, tb_writer=None, **kwargs):
        """
        Training step with aggressive memory management.
        """
        n = batch['input_ids'].size(0)
        start_time = time.perf_counter()
        log_dict = kwargs['log']

        # Memory before
        mem_before = torch.cuda.memory_allocated() / 1024**3

        # Execute AdaSmooth step
        losses, weights, _ = self.adasmooth_step(model, batch)

        # Memory after
        mem_after = torch.cuda.memory_allocated() / 1024**3

        # Update step counter
        self.state.global_training_steps += 1

        # Logging (prevent unbounded list growth)
        total_time = time.perf_counter() - start_time

        # Limit history lists to prevent memory leak
        if len(self.state.time) > 100:
            self.state.time = self.state.time[-50:]
        if len(self.state.query) > 100:
            self.state.query = self.state.query[-50:]

        self.state.time.append(total_time)
        self.state.query.append(self.K * n)

        # Compute metrics (all as Python floats)
        beta_t = self.beta_init / (1 + self.beta_decay * (self.state.global_training_steps - 1))

        w_array = np.array(weights, dtype=np.float32)
        ess = float(1.0 / (w_array ** 2).sum())
        entropy = float(-(w_array * np.log(w_array + 1e-10)).sum())

        log_dict['train_loss'] = float(np.mean(losses))
        log_dict['time'] = float(total_time)
        log_dict['beta_t'] = float(beta_t)
        log_dict['max_weight'] = float(max(weights))
        log_dict['min_weight'] = float(min(weights))
        log_dict['ess'] = ess
        log_dict['entropy'] = entropy
        log_dict['mem_before'] = float(mem_before)
        log_dict['mem_after'] = float(mem_after)
        log_dict['mem_delta'] = float(mem_after - mem_before)

        # Force cleanup after every training step
        gc.collect()
        if self.state.global_training_steps % 10 == 0:
            torch.cuda.empty_cache()

        return log_dict
