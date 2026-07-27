import math
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from common.logging import logger as logging

from .glue_transformer_mezo import GLUETransformerMeZo


class GLUETransformerAGZO(GLUETransformerMeZo):
    """Activation-Guided Zeroth-Order optimization for ENGINE."""

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
        rank: int = 1,
        weight_decay: float = 0.01,
        num_queries: int = 2,
        agzo_power_iter_steps: int = 5,
        **kwargs,
    ):
        if not full_parameter:
            raise NotImplementedError(
                "AGZO currently supports full-parameter optimization only."
            )
        if rank <= 0:
            raise ValueError("AGZO rank must be positive.")
        if agzo_power_iter_steps <= 0:
            raise ValueError("AGZO agzo_power_iter_steps must be positive.")
        if zero_order_eps <= 0:
            raise ValueError("AGZO perturbation_scale must be positive.")
        if num_queries < 2 or num_queries % 2 != 0:
            raise ValueError("AGZO num_queries must be a positive even number >= 2.")
        if weight_decay < 0:
            raise ValueError("AGZO weight_decay must be non-negative.")

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
            rank=rank,
            weight_decay=weight_decay,
            agzo_power_iter_steps=agzo_power_iter_steps,
            **kwargs,
        )

        self.rank = int(rank)
        self.weight_decay = float(weight_decay)
        self.agzo_power_iter_steps = int(agzo_power_iter_steps)

        self._param_index = {}
        self._params_by_name = {}
        self._agzo_module_specs = {}
        self._agzo_basis = {}
        self._agzo_hook_handles = []
        self._agzo_hooks_registered = False
        self._agzo_collecting = False
        self._agzo_basis_seed = None
        self._basis_failure_warned = False
        self._no_guided_module_warned = False

        self._basis_collection_count = 0
        self._basis_memory_gb = 0.0
        self._effective_rank_min = 0
        self._effective_rank_max = 0
        self._guided_parameter_count = 0
        self._fallback_parameter_count = 0
        self._guided_parameter_numel = 0
        self._fallback_parameter_numel = 0
        self._guided_parameter_ratio = 0.0

    def configure_params(self):
        super().configure_params()
        self.params = [
            (name, param)
            for name, param in self.params
            if param.requires_grad
        ]
        if not self.params:
            raise ValueError("AGZO found no trainable parameters.")

        self._param_index = {
            name: param_index
            for param_index, (name, _) in enumerate(self.params)
        }
        self._params_by_name = dict(self.params)
        self._register_agzo_hooks()

        if not hasattr(self.state, "num_forward_passes"):
            self.state.num_forward_passes = 0

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

    def _register_agzo_hooks(self):
        if self._agzo_hooks_registered:
            return

        named_parameters = dict(self.model.named_parameters())
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layout = "out_in"
                input_dim = int(module.in_features)
                output_dim = int(module.out_features)
            elif isinstance(module, Conv1D):
                layout = "in_out"
                input_dim = int(module.weight.shape[0])
                output_dim = int(module.weight.shape[1])
            else:
                continue

            param_name = f"{module_name}.weight" if module_name else "weight"
            param = named_parameters.get(param_name)
            if (
                param is None
                or param is not module.weight
                or not param.requires_grad
                or param.ndim != 2
                or param_name not in self._param_index
                or param_name in self._agzo_module_specs
            ):
                continue

            expected_shape = (
                (output_dim, input_dim)
                if layout == "out_in"
                else (input_dim, output_dim)
            )
            if tuple(param.shape) != expected_shape:
                continue

            self._agzo_module_specs[param_name] = {
                "layout": layout,
                "input_dim": input_dim,
                "output_dim": output_dim,
            }
            handle = module.register_forward_hook(
                self._make_agzo_hook(
                    param_name=param_name,
                    layout=layout,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
            )
            self._agzo_hook_handles.append(handle)

        self._agzo_hooks_registered = True
        if not self._agzo_module_specs and not self._no_guided_module_warned:
            logging.warning(
                "AGZO registered no supported trainable Linear/Conv1D weights; "
                "all parameters will use dense Gaussian fallback."
            )
            self._no_guided_module_warned = True

    def _make_agzo_hook(
        self,
        param_name: str,
        layout: str,
        input_dim: int,
        output_dim: int,
    ):
        def hook(module, inputs, output):
            del module, output
            if not self._agzo_collecting or self._agzo_basis_seed is None:
                return

            activation = next(
                (value for value in inputs if isinstance(value, torch.Tensor)),
                None,
            )
            if (
                activation is None
                or activation.ndim == 0
                or activation.shape[-1] != input_dim
            ):
                return

            param = self._params_by_name[param_name]
            param_index = self._param_index[param_name]
            basis = self._compute_activation_basis(
                activation=activation,
                param=param,
                param_index=param_index,
                layout=layout,
                input_dim=input_dim,
                output_dim=output_dim,
                basis_seed=self._agzo_basis_seed,
            )
            if basis is not None:
                self._agzo_basis[param_name] = basis

        return hook

    @staticmethod
    def _block_power_iteration(
        h_2d: torch.Tensor,
        rank: int,
        num_steps: int,
        generator: torch.Generator,
    ):
        if h_2d.ndim != 2 or h_2d.numel() == 0:
            return None

        input_dim = int(h_2d.shape[1])
        if rank <= 0 or rank > min(h_2d.shape):
            return None

        q = torch.randn(
            (input_dim, rank),
            device=h_2d.device,
            dtype=torch.float32,
            generator=generator,
        )
        q = torch.linalg.qr(q, mode="reduced").Q

        for _ in range(num_steps):
            y = h_2d @ q
            z = h_2d.transpose(0, 1) @ y
            if not torch.isfinite(z).all():
                return None
            q = torch.linalg.qr(z, mode="reduced").Q
            if not torch.isfinite(q).all():
                return None

        return q.transpose(0, 1).contiguous()

    def _compute_activation_basis(
        self,
        activation: torch.Tensor,
        param: torch.nn.Parameter,
        param_index: int,
        layout: str,
        input_dim: int,
        output_dim: int,
        basis_seed: int,
    ):
        if activation.numel() == 0 or activation.shape[-1] != input_dim:
            return None

        h_2d = activation.detach().reshape(-1, input_dim).to(torch.float32)
        effective_rank = min(
            self.rank,
            int(h_2d.shape[0]),
            input_dim,
            output_dim,
        )
        if effective_rank <= 0:
            return None

        generator = self._make_generator(
            h_2d.device,
            basis_seed,
            param_index,
            stream=0,
        )
        try:
            basis_t = self._block_power_iteration(
                h_2d=h_2d,
                rank=effective_rank,
                num_steps=self.agzo_power_iter_steps,
                generator=generator,
            )
        except RuntimeError as error:
            if not self._basis_failure_warned:
                logging.warning(
                    f"AGZO activation-basis computation failed; using dense "
                    f"fallback for affected weights. First error: {error}"
                )
                self._basis_failure_warned = True
            return None

        if basis_t is None or not torch.isfinite(basis_t).all():
            return None

        return {
            "basis_t": basis_t.to(device=param.device, dtype=torch.float32),
            "layout": layout,
            "effective_rank": effective_rank,
            "input_dim": input_dim,
            "output_dim": output_dim,
        }

    def _update_basis_statistics(self):
        guided_names = set(self._agzo_basis)
        effective_ranks = [
            int(spec["effective_rank"])
            for spec in self._agzo_basis.values()
        ]

        guided_numel = 0
        fallback_numel = 0
        basis_bytes = 0
        for name, param in self.params:
            if name in guided_names:
                guided_numel += param.numel()
                basis_t = self._agzo_basis[name]["basis_t"]
                basis_bytes += basis_t.numel() * basis_t.element_size()
            else:
                fallback_numel += param.numel()

        total_numel = guided_numel + fallback_numel
        self._guided_parameter_count = len(guided_names)
        self._fallback_parameter_count = len(self.params) - len(guided_names)
        self._guided_parameter_numel = guided_numel
        self._fallback_parameter_numel = fallback_numel
        self._guided_parameter_ratio = (
            guided_numel / total_numel if total_numel else 0.0
        )
        self._basis_memory_gb = basis_bytes / (1024**3)
        self._effective_rank_min = min(effective_ranks, default=0)
        self._effective_rank_max = max(effective_ranks, default=0)

    @torch.no_grad()
    def _collect_activation_basis(
        self,
        model,
        batch,
        basis_seed: int,
    ):
        batch_size = int(batch["input_ids"].shape[0])
        if batch_size > self.batchsize_limit:
            raise ValueError(
                "AGZO activation collection currently requires "
                "batch_size <= batchsize_limit."
            )
        if not self._agzo_hooks_registered:
            self._register_agzo_hooks()

        self._agzo_basis.clear()
        self._agzo_basis_seed = int(basis_seed)
        self._agzo_collecting = True
        start_time = time.perf_counter()
        try:
            loss = self.zo_forward_memory_eff(model, batch)
        finally:
            self._agzo_collecting = False
            self._agzo_basis_seed = None

        self._basis_collection_count += 1
        self._update_basis_statistics()
        return loss, time.perf_counter() - start_time

    def _make_direction(
        self,
        name: str,
        param: torch.nn.Parameter,
        param_index: int,
        direction_seed: int,
    ):
        basis = self._agzo_basis.get(name)
        if basis is None:
            generator = self._make_generator(
                param.device,
                direction_seed,
                param_index,
                stream=2,
            )
            direction = torch.randn(
                param.shape,
                device=param.device,
                dtype=param.dtype,
                generator=generator,
            )
            direction.mul_(self.z_std)
            return direction

        effective_rank = int(basis["effective_rank"])
        generator = self._make_generator(
            param.device,
            direction_seed,
            param_index,
            stream=1,
        )
        random_factor = torch.randn(
            (basis["output_dim"], effective_rank),
            device=param.device,
            dtype=torch.float32,
            generator=generator,
        )
        random_factor.mul_(self.z_std)
        basis_t = basis["basis_t"]

        if basis["layout"] == "out_in":
            direction = random_factor @ basis_t
        elif basis["layout"] == "in_out":
            direction = basis_t.transpose(0, 1) @ random_factor.transpose(0, 1)
        else:
            raise RuntimeError(f"Unknown AGZO layout: {basis['layout']}")

        direction.div_(math.sqrt(effective_rank))
        direction = direction.to(dtype=param.dtype)
        if tuple(direction.shape) != tuple(param.shape):
            raise RuntimeError(
                f"AGZO direction shape {tuple(direction.shape)} does not match "
                f"parameter {name} shape {tuple(param.shape)}."
            )
        return direction

    @torch.no_grad()
    def _perturb_parameters(
        self,
        direction_seed: int,
        scaling_factor: float,
    ):
        alpha = scaling_factor * self.zero_order_eps
        for param_index, (name, param) in enumerate(self.params):
            direction = self._make_direction(
                name,
                param,
                param_index,
                direction_seed,
            )
            param.add_(direction, alpha=alpha)

    @torch.no_grad()
    def _central_difference(
        self,
        model,
        batch,
        direction_seed: int,
    ):
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
    def _apply_agzo_update(self, direction_seeds, projected_grads):
        learning_rate = self.state.learning_rate
        if self.weight_decay:
            decay = 1.0 - learning_rate * self.weight_decay
            for _, param in self.params:
                param.mul_(decay)

        scale = -learning_rate / len(direction_seeds)
        for direction_seed, projected_grad in zip(
            direction_seeds,
            projected_grads,
        ):
            for param_index, (name, param) in enumerate(self.params):
                direction = self._make_direction(
                    name,
                    param,
                    param_index,
                    direction_seed,
                )
                param.add_(direction, alpha=scale * projected_grad)

    @torch.no_grad()
    def training_step(self, model, batch, tb_writer=None, **kwargs):
        del tb_writer
        start_time = time.perf_counter()
        log_dict = kwargs["log"]
        basis_seed = int(np.random.randint(1_000_000_000))

        try:
            loss_unperturbed, basis_collection_time = (
                self._collect_activation_basis(model, batch, basis_seed)
            )

            num_directions = self.num_queries // 2
            direction_seeds = [
                int(np.random.randint(1_000_000_000))
                for _ in range(num_directions)
            ]
            projected_grads = []
            for direction_seed in direction_seeds:
                _, projected_grad = self._central_difference(
                    model,
                    batch,
                    direction_seed,
                )
                projected_grads.append(projected_grad)

            self._apply_agzo_update(direction_seeds, projected_grads)

            activation_forward_evals = 1
            probe_forward_evals = self.num_queries
            total_forward_evals = (
                activation_forward_evals + probe_forward_evals
            )
            batch_size = int(batch["input_ids"].shape[0])

            self.state.global_training_steps += 1
            self.state.query.append(batch_size * total_forward_evals)
            self.state.num_forward_passes += total_forward_evals
            total_time = time.perf_counter() - start_time
            self.state.time.append(total_time)

            projected_grad_mean = float(np.mean(projected_grads))
            projected_grad_abs_mean = float(
                np.mean(np.abs(projected_grads))
            )
            self.state.proj_val.append(projected_grad_abs_mean)

            log_dict["train_loss"] = float(loss_unperturbed.item())
            log_dict["time"] = total_time
            log_dict["num_forward_passes"] = (
                self.state.num_forward_passes
            )
            log_dict["agzo_num_directions"] = num_directions
            log_dict["agzo_projected_grad_mean"] = projected_grad_mean
            log_dict["agzo_projected_grad_abs_mean"] = (
                projected_grad_abs_mean
            )
            log_dict["agzo_rank_requested"] = self.rank
            log_dict["agzo_effective_rank_min"] = (
                self._effective_rank_min
            )
            log_dict["agzo_effective_rank_max"] = (
                self._effective_rank_max
            )
            log_dict["agzo_power_iter_steps"] = (
                self.agzo_power_iter_steps
            )
            log_dict["agzo_basis_collection_count"] = (
                self._basis_collection_count
            )
            log_dict["agzo_basis_collection_time_sec"] = (
                basis_collection_time
            )
            log_dict["agzo_basis_memory_gb"] = self._basis_memory_gb
            log_dict["agzo_guided_parameter_count"] = (
                self._guided_parameter_count
            )
            log_dict["agzo_fallback_parameter_count"] = (
                self._fallback_parameter_count
            )
            log_dict["agzo_guided_parameter_numel"] = (
                self._guided_parameter_numel
            )
            log_dict["agzo_fallback_parameter_numel"] = (
                self._fallback_parameter_numel
            )
            log_dict["agzo_guided_parameter_ratio"] = (
                self._guided_parameter_ratio
            )
            log_dict["agzo_activation_forward_evals"] = (
                activation_forward_evals
            )
            log_dict["agzo_probe_forward_evals"] = probe_forward_evals
            log_dict["agzo_total_forward_evals"] = total_forward_evals
            log_dict["agzo_official_q1"] = int(self.num_queries == 2)
        finally:
            self._agzo_collecting = False
            self._agzo_basis_seed = None
            self._agzo_basis.clear()
