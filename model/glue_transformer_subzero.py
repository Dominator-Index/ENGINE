import math
import time
from typing import Optional

import numpy as np
import torch

from common.logging import logger as logging

from .glue_transformer_mezo import GLUETransformerMeZo


class GLUETransformerSubZero(GLUETransformerMeZo):
    """SubZero with the same training interface as the existing ZO methods."""

    _MAX_SEED = 2**63 - 1

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
        logger_type: str = "wandb",
        hf_token: Optional[str] = None,
        lora: bool = False,
        model_init_seed: Optional[int] = None,
        step_interval: int = 50,
        rank: int = 2,
        weight_decay: float = 0.01,
        num_queries: int = 2,
        **kwargs,
    ):
        if not full_parameter:
            raise NotImplementedError(
                "SubZero currently supports full-parameter optimization only."
            )
        if rank <= 0:
            raise ValueError("SubZero rank must be positive.")
        if step_interval <= 0:
            raise ValueError("SubZero step_interval must be positive.")
        if zero_order_eps <= 0:
            raise ValueError("SubZero perturbation_scale must be positive.")
        if num_queries < 2 or num_queries % 2 != 0:
            raise ValueError("SubZero num_queries must be a positive even number >= 2.")
        if weight_decay < 0:
            raise ValueError("SubZero weight_decay must be non-negative.")

        super().__init__(
            model_name_or_path=model_name_or_path,
            num_labels=num_labels,
            zero_order_eps=zero_order_eps,
            learning_rate_aux=learning_rate_aux,
            minibatch=minibatch,
            max_norm=max_norm,
            z_std=z_std,
            lr_anneal=lr_anneal,
            full_parameter=full_parameter,
            batchsize_limit=batchsize_limit,
            eval_splits=eval_splits,
            soft_prompt=soft_prompt,
            use_SGD=use_SGD,
            logger_type=logger_type,
            hf_token=hf_token,
            lora=lora,
            model_init_seed=model_init_seed,
            num_queries=num_queries,
            **kwargs,
        )

        self.rank = rank
        self.basis_upd_interval = step_interval
        self.weight_decay = weight_decay
        self.subspace_basis = {}
        self.basis_refresh_count = 0
        self.last_basis_refresh_step = -1
        self.last_basis_seed = None
        self._basis_memory_gb = 0.0
        self._effective_dim = 0
        self._effective_rank_min = 0
        self._effective_rank_max = 0
        self._rank_clamp_warned = False
        self.state.num_forward_passes = 0

    def configure_params(self):
        super().configure_params()
        self.params = [(name, param) for name, param in self.params if param.requires_grad]
        if not self.params:
            raise ValueError("SubZero found no trainable parameters.")
        if not hasattr(self.state, "num_forward_passes"):
            self.state.num_forward_passes = 0

    @staticmethod
    def _matrix_shape(param):
        shape = tuple(param.squeeze().shape)
        if len(shape) != 2:
            return None
        return int(shape[0]), int(shape[1])

    @classmethod
    def _mix_seed(cls, seed: int, param_index: int, stream: int) -> int:
        mixed = (
            int(seed)
            + 1_000_003 * (param_index + 1)
            + 97_003 * (stream + 1)
        )
        return mixed % cls._MAX_SEED

    @classmethod
    def _make_generator(
        cls,
        device: torch.device,
        seed: int,
        param_index: int,
        stream: int,
    ):
        generator = torch.Generator(device=device)
        generator.manual_seed(cls._mix_seed(seed, param_index, stream))
        return generator

    @torch.no_grad()
    def _refresh_subspace_basis(self, refresh_seed: Optional[int] = None):
        start_time = time.perf_counter()
        if refresh_seed is None:
            refresh_seed = int(np.random.randint(1_000_000_000))

        new_basis = {}
        effective_ranks = []
        effective_dim = 0
        basis_bytes = 0

        for param_index, (name, param) in enumerate(self.params):
            matrix_shape = self._matrix_shape(param)
            if matrix_shape is None:
                effective_dim += param.numel()
                continue

            m, n = matrix_shape
            effective_rank = min(self.rank, m, n)
            if effective_rank < self.rank and not self._rank_clamp_warned:
                logging.warning(
                    "SubZero rank %s is larger than matrix shape %s for %s; "
                    "using effective rank %s.",
                    self.rank,
                    matrix_shape,
                    name,
                    effective_rank,
                )
                self._rank_clamp_warned = True

            u_generator = self._make_generator(
                param.device, refresh_seed, param_index, stream=0
            )
            v_generator = self._make_generator(
                param.device, refresh_seed, param_index, stream=1
            )
            u_random = torch.randn(
                (m, effective_rank),
                device=param.device,
                dtype=torch.float32,
                generator=u_generator,
            )
            v_random = torch.randn(
                (n, effective_rank),
                device=param.device,
                dtype=torch.float32,
                generator=v_generator,
            )
            u = torch.linalg.qr(u_random, mode="reduced").Q.to(param.dtype).contiguous()
            vt = (
                torch.linalg.qr(v_random, mode="reduced")
                .Q.to(param.dtype)
                .t()
                .contiguous()
            )
            new_basis[name] = {
                "U": u,
                "Vt": vt,
                "matrix_shape": matrix_shape,
                "effective_rank": effective_rank,
            }
            effective_ranks.append(effective_rank)
            effective_dim += effective_rank**2
            basis_bytes += u.numel() * u.element_size()
            basis_bytes += vt.numel() * vt.element_size()

        self.subspace_basis = new_basis
        self.basis_refresh_count += 1
        self.last_basis_refresh_step = self.state.global_training_steps
        self.last_basis_seed = refresh_seed
        self._basis_memory_gb = basis_bytes / (1024**3)
        self._effective_dim = effective_dim
        self._effective_rank_min = min(effective_ranks, default=0)
        self._effective_rank_max = max(effective_ranks, default=0)
        return time.perf_counter() - start_time

    def _make_direction(self, name, param, param_index: int, direction_seed: int):
        basis = self.subspace_basis.get(name)
        if basis is None:
            generator = self._make_generator(
                param.device, direction_seed, param_index, stream=3
            )
            return torch.randn(
                param.shape,
                device=param.device,
                dtype=param.dtype,
                generator=generator,
            ) * self.z_std

        effective_rank = basis["effective_rank"]
        generator = self._make_generator(
            param.device, direction_seed, param_index, stream=2
        )
        core = torch.randn(
            (effective_rank, effective_rank),
            device=param.device,
            dtype=param.dtype,
            generator=generator,
        ) * self.z_std
        direction = basis["U"] @ core @ basis["Vt"]
        direction.mul_(math.sqrt(param.numel() / core.numel()))
        return direction.reshape(param.shape)

    @torch.no_grad()
    def _perturb_parameters(self, direction_seed: int, scaling_factor: float):
        alpha = scaling_factor * self.zero_order_eps
        for param_index, (name, param) in enumerate(self.params):
            direction = self._make_direction(
                name, param, param_index, direction_seed
            )
            param.add_(direction, alpha=alpha)

    @torch.no_grad()
    def _central_difference(self, model, batch, direction_seed: int):
        position = 0
        try:
            self._perturb_parameters(direction_seed, scaling_factor=1.0)
            position = 1
            loss_plus = self.zo_forward_memory_eff(model, batch)

            self._perturb_parameters(direction_seed, scaling_factor=-2.0)
            position = -1
            loss_minus = self.zo_forward_memory_eff(model, batch)
        finally:
            if position == 1:
                self._perturb_parameters(direction_seed, scaling_factor=-1.0)
            elif position == -1:
                self._perturb_parameters(direction_seed, scaling_factor=1.0)

        projected_grad = (
            (loss_plus - loss_minus) / (2.0 * self.zero_order_eps)
        ).item()
        return loss_plus.item(), projected_grad

    @torch.no_grad()
    def _apply_subzero_update(self, direction_seeds, projected_grads):
        learning_rate = self.state.learning_rate
        if self.weight_decay:
            decay = 1.0 - learning_rate * self.weight_decay
            for _, param in self.params:
                param.mul_(decay)

        scale = -learning_rate / len(direction_seeds)
        for direction_seed, projected_grad in zip(
            direction_seeds, projected_grads
        ):
            for param_index, (name, param) in enumerate(self.params):
                direction = self._make_direction(
                    name, param, param_index, direction_seed
                )
                param.add_(direction, alpha=scale * projected_grad)

    @torch.no_grad()
    def training_step(self, model, batch, tb_writer=None, **kwargs):
        del tb_writer
        start_time = time.perf_counter()
        log_dict = kwargs["log"]
        step = self.state.global_training_steps

        basis_refreshed = step % self.basis_upd_interval == 0
        basis_refresh_time = 0.0
        if basis_refreshed:
            basis_refresh_time = self._refresh_subspace_basis()

        num_directions = self.num_queries // 2
        direction_seeds = [
            int(np.random.randint(1_000_000_000)) for _ in range(num_directions)
        ]
        losses = []
        projected_grads = []
        for direction_seed in direction_seeds:
            loss, projected_grad = self._central_difference(
                model, batch, direction_seed
            )
            losses.append(loss)
            projected_grads.append(projected_grad)

        self._apply_subzero_update(direction_seeds, projected_grads)

        self.state.global_training_steps += 1
        self.state.query.append(self.num_queries)
        self.state.num_forward_passes += self.num_queries
        total_time = time.perf_counter() - start_time
        self.state.time.append(total_time)

        projected_grad_mean = float(np.mean(projected_grads))
        projected_grad_abs_mean = float(np.mean(np.abs(projected_grads)))
        self.state.proj_val.append(projected_grad_abs_mean)

        log_dict["train_loss"] = float(np.mean(losses))
        log_dict["time"] = total_time
        log_dict["num_forward_passes"] = self.state.num_forward_passes
        log_dict["subzero_num_directions"] = num_directions
        log_dict["subzero_projected_grad_mean"] = projected_grad_mean
        log_dict["subzero_projected_grad_abs_mean"] = projected_grad_abs_mean
        log_dict["subzero_basis_refreshed"] = int(basis_refreshed)
        log_dict["subzero_refresh_count"] = self.basis_refresh_count
        log_dict["subzero_basis_refresh_time_sec"] = basis_refresh_time
        log_dict["subzero_rank_requested"] = self.rank
        log_dict["subzero_effective_rank_min"] = self._effective_rank_min
        log_dict["subzero_effective_rank_max"] = self._effective_rank_max
        log_dict["subzero_effective_dim"] = self._effective_dim
        log_dict["subzero_basis_memory_gb"] = self._basis_memory_gb
        log_dict["subzero_official_q1"] = int(self.num_queries == 2)
