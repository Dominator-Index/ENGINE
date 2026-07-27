import time
from typing import Optional

import numpy as np
import torch
from transformers.pytorch_utils import Conv1D

from common.logging import logger as logging

from .glue_transformer_mezo import GLUETransformerMeZo


def zeropower_via_newtonschulz5(matrix: torch.Tensor) -> torch.Tensor:
    """Approximate the matrix sign used by the official ZO-Muon code."""
    if matrix.ndim != 2:
        raise ValueError("ZO-Muon Newton-Schulz input must be a matrix.")

    a, b, c = (3.4445, -4.7750, 2.0315)
    transpose_needed = matrix.size(-2) > matrix.size(-1)
    x = matrix.mT if transpose_needed else matrix
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(5):
        gram = x @ x.mT
        polynomial = b * gram + c * gram @ gram
        x = a * x + polynomial @ x

    return x.mT if transpose_needed else x


class GLUETransformerZOMuon(GLUETransformerMeZo):
    """ZO-Muon with ENGINE's existing full-parameter ZO interface."""

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
        num_queries: int = 10,
        zomuon_nonmatrix_lr: float = 1e-7,
        **kwargs,
    ):
        if not full_parameter:
            raise NotImplementedError(
                "ZO-Muon currently supports full-parameter optimization only."
            )
        if rank <= 0:
            raise ValueError("ZO-Muon rank must be positive.")
        if step_interval <= 0:
            raise ValueError("ZO-Muon step_interval must be positive.")
        if zero_order_eps <= 0:
            raise ValueError("ZO-Muon perturbation_scale must be positive.")
        if num_queries < 3:
            raise ValueError(
                "ZO-Muon num_queries must be >= 3: one baseline and at "
                "least two perturbed evaluations."
            )
        if weight_decay < 0:
            raise ValueError("ZO-Muon weight_decay must be non-negative.")
        if zomuon_nonmatrix_lr <= 0:
            raise ValueError("ZO-Muon zomuon_nonmatrix_lr must be positive.")

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
            step_interval=step_interval,
            rank=rank,
            weight_decay=weight_decay,
            zomuon_nonmatrix_lr=zomuon_nonmatrix_lr,
            **kwargs,
        )

        self.rank = int(rank)
        self.step_interval = int(step_interval)
        self.weight_decay = float(weight_decay)
        self.zomuon_nonmatrix_lr = float(zomuon_nonmatrix_lr)

        self.parameter_specs = {}
        self.projection_matrices = {}
        self.basis_refresh_count = 0
        self.last_basis_refresh_step = -1
        self.last_basis_seed = None
        self._basis_memory_gb = 0.0
        self._effective_dim = 0
        self._effective_rank_min = 0
        self._effective_rank_max = 0
        self._matrix_parameter_count = 0
        self._nonmatrix_parameter_count = 0
        self._rank_clamp_warned = False
        self.state.num_forward_passes = 0

    def configure_params(self):
        super().configure_params()
        self.params = [
            (name, param)
            for name, param in self.params
            if param.requires_grad
        ]
        if not self.params:
            raise ValueError("ZO-Muon found no trainable parameters.")

        self._build_parameter_specs()
        if self._matrix_parameter_count == 0:
            raise ValueError("ZO-Muon found no trainable matrix parameters.")

        # Layouts are rebuilt here, so stale projections cannot survive a
        # second configure call or a resumed run.
        self.projection_matrices = {}
        if not hasattr(self.state, "num_forward_passes"):
            self.state.num_forward_passes = 0

    def _build_parameter_specs(self):
        conv1d_weight_ids = {
            id(module.weight)
            for module in self.model.modules()
            if isinstance(module, Conv1D)
        }

        specs = {}
        effective_ranks = []
        effective_dim = 0
        matrix_count = 0
        nonmatrix_count = 0

        for name, param in self.params:
            if param.ndim != 2:
                specs[name] = {"kind": "nonmatrix"}
                effective_dim += param.numel()
                nonmatrix_count += 1
                continue

            if id(param) in conv1d_weight_ids:
                layout = "conv1d_transposed"
                logical_shape = (int(param.shape[1]), int(param.shape[0]))
            else:
                layout = "normal"
                logical_shape = (int(param.shape[0]), int(param.shape[1]))

            m, n = logical_shape
            effective_rank = min(self.rank, m, n)
            if effective_rank <= 0:
                raise ValueError(
                    f"ZO-Muon cannot optimize empty matrix parameter {name} "
                    f"with shape {tuple(param.shape)}."
                )
            if effective_rank < self.rank and not self._rank_clamp_warned:
                logging.warning(
                    "ZO-Muon rank %s is larger than logical matrix shape %s "
                    "for %s; using effective rank %s.",
                    self.rank,
                    logical_shape,
                    name,
                    effective_rank,
                )
                self._rank_clamp_warned = True

            specs[name] = {
                "kind": "matrix",
                "layout": layout,
                "logical_shape": logical_shape,
                "effective_rank": effective_rank,
            }
            effective_ranks.append(effective_rank)
            effective_dim += effective_rank * n
            matrix_count += 1

        self.parameter_specs = specs
        self._effective_dim = effective_dim
        self._effective_rank_min = min(effective_ranks, default=0)
        self._effective_rank_max = max(effective_ranks, default=0)
        self._matrix_parameter_count = matrix_count
        self._nonmatrix_parameter_count = nonmatrix_count

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
    def _refresh_projection_matrices(
        self,
        refresh_seed: Optional[int] = None,
    ):
        start_time = time.perf_counter()
        if refresh_seed is None:
            refresh_seed = int(np.random.randint(1_000_000_000))

        new_projections = {}
        basis_bytes = 0

        for param_index, (name, param) in enumerate(self.params):
            spec = self.parameter_specs[name]
            if spec["kind"] != "matrix":
                continue

            m, _ = spec["logical_shape"]
            effective_rank = spec["effective_rank"]
            generator = self._make_generator(
                param.device,
                refresh_seed,
                param_index,
                stream=0,
            )
            random_matrix = torch.randn(
                (m, effective_rank),
                device=param.device,
                dtype=torch.float32,
                generator=generator,
            )
            projection, triangular = torch.linalg.qr(
                random_matrix,
                mode="reduced",
            )
            signs = torch.sign(torch.diagonal(triangular))
            signs[signs == 0] = 1.0
            projection.mul_(signs)
            projection = projection.to(param.dtype).contiguous()

            new_projections[name] = projection
            basis_bytes += projection.numel() * projection.element_size()

        self.projection_matrices = new_projections
        self.basis_refresh_count += 1
        self.last_basis_refresh_step = self.state.global_training_steps
        self.last_basis_seed = int(refresh_seed)
        self._basis_memory_gb = basis_bytes / (1024**3)
        return time.perf_counter() - start_time

    def _sample_matrix_psi(
        self,
        param: torch.nn.Parameter,
        param_index: int,
        direction_seed: int,
        effective_rank: int,
        n: int,
    ):
        generator = self._make_generator(
            param.device,
            direction_seed,
            param_index,
            stream=1,
        )
        psi = torch.randn(
            (effective_rank, n),
            device=param.device,
            dtype=param.dtype,
            generator=generator,
        )
        psi.mul_(self.z_std)
        return psi

    def _sample_nonmatrix_z(
        self,
        param: torch.nn.Parameter,
        param_index: int,
        direction_seed: int,
    ):
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

    def _make_direction(
        self,
        name: str,
        param: torch.nn.Parameter,
        param_index: int,
        direction_seed: int,
    ):
        spec = self.parameter_specs[name]
        if spec["kind"] != "matrix":
            return self._sample_nonmatrix_z(
                param,
                param_index,
                direction_seed,
            )

        projection = self.projection_matrices.get(name)
        if projection is None:
            raise RuntimeError(
                f"ZO-Muon projection for matrix parameter {name} is missing."
            )

        _, n = spec["logical_shape"]
        psi = self._sample_matrix_psi(
            param,
            param_index,
            direction_seed,
            spec["effective_rank"],
            n,
        )
        logical_direction = projection @ psi
        if spec["layout"] == "conv1d_transposed":
            return logical_direction.t().contiguous()
        return logical_direction

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
    def _evaluate_direction(
        self,
        model,
        batch,
        baseline_loss: torch.Tensor,
        direction_seed: int,
    ):
        perturbed = False
        try:
            self._perturb_parameters(
                direction_seed,
                scaling_factor=1.0,
            )
            perturbed = True
            loss_plus = self.zo_forward_memory_eff(model, batch)
        finally:
            if perturbed:
                self._perturb_parameters(
                    direction_seed,
                    scaling_factor=-1.0,
                )

        projected_scalar = (
            (loss_plus - baseline_loss) / self.zero_order_eps
        ).item()
        return loss_plus.item(), projected_scalar

    @torch.no_grad()
    def _apply_zomuon_update(
        self,
        direction_seeds,
        projected_scalars,
    ):
        if not direction_seeds:
            raise ValueError("ZO-Muon requires at least one direction.")
        if len(direction_seeds) != len(projected_scalars):
            raise ValueError(
                "ZO-Muon direction seeds and projected scalars must match."
            )

        matrix_lr = self.state.learning_rate
        inverse_num_directions = 1.0 / len(direction_seeds)

        for param_index, (name, param) in enumerate(self.params):
            spec = self.parameter_specs[name]

            if spec["kind"] == "matrix":
                projection = self.projection_matrices.get(name)
                if projection is None:
                    raise RuntimeError(
                        f"ZO-Muon projection for matrix parameter {name} "
                        "is missing."
                    )

                _, n = spec["logical_shape"]
                effective_rank = spec["effective_rank"]
                lowdim_rge = torch.zeros(
                    (effective_rank, n),
                    device=param.device,
                    dtype=param.dtype,
                )
                for direction_seed, projected_scalar in zip(
                    direction_seeds,
                    projected_scalars,
                ):
                    psi = self._sample_matrix_psi(
                        param,
                        param_index,
                        direction_seed,
                        effective_rank,
                        n,
                    )
                    lowdim_rge.add_(
                        psi,
                        alpha=projected_scalar * inverse_num_directions,
                    )

                lowdim_update = zeropower_via_newtonschulz5(lowdim_rge)
                logical_update = projection @ lowdim_update
                update = (
                    logical_update.t().contiguous()
                    if spec["layout"] == "conv1d_transposed"
                    else logical_update
                )

                if self.weight_decay:
                    param.mul_(1.0 - matrix_lr * self.weight_decay)
                param.add_(update, alpha=-matrix_lr)
                continue

            grad_estimate = torch.zeros_like(param)
            for direction_seed, projected_scalar in zip(
                direction_seeds,
                projected_scalars,
            ):
                direction = self._sample_nonmatrix_z(
                    param,
                    param_index,
                    direction_seed,
                )
                grad_estimate.add_(
                    direction,
                    alpha=projected_scalar * inverse_num_directions,
                )

            if self.weight_decay:
                param.mul_(
                    1.0 - self.zomuon_nonmatrix_lr * self.weight_decay
                )
            param.add_(
                grad_estimate,
                alpha=-self.zomuon_nonmatrix_lr,
            )

    @torch.no_grad()
    def training_step(self, model, batch, tb_writer=None, **kwargs):
        del tb_writer
        start_time = time.perf_counter()
        log_dict = kwargs["log"]
        step = self.state.global_training_steps

        basis_refreshed = (
            not self.projection_matrices
            or step % self.step_interval == 0
        )
        basis_refresh_time = 0.0
        if basis_refreshed:
            basis_refresh_time = self._refresh_projection_matrices()

        baseline_loss = self.zo_forward_memory_eff(model, batch)
        num_directions = self.num_queries - 1
        direction_seeds = [
            int(np.random.randint(1_000_000_000))
            for _ in range(num_directions)
        ]

        perturbed_losses = []
        projected_scalars = []
        for direction_seed in direction_seeds:
            loss_plus, projected_scalar = self._evaluate_direction(
                model,
                batch,
                baseline_loss,
                direction_seed,
            )
            perturbed_losses.append(loss_plus)
            projected_scalars.append(projected_scalar)

        self._apply_zomuon_update(
            direction_seeds,
            projected_scalars,
        )

        self.state.global_training_steps += 1
        self.state.query.append(self.num_queries)
        self.state.num_forward_passes += self.num_queries
        total_time = time.perf_counter() - start_time
        self.state.time.append(total_time)

        projected_scalar_mean = float(np.mean(projected_scalars))
        projected_scalar_abs_mean = float(
            np.mean(np.abs(projected_scalars))
        )
        self.state.proj_val.append(projected_scalar_abs_mean)

        log_dict["train_loss"] = baseline_loss.item()
        log_dict["time"] = total_time
        log_dict["num_forward_passes"] = self.state.num_forward_passes
        log_dict["zomuon_num_directions"] = num_directions
        log_dict["zomuon_baseline_forward_evals"] = 1
        log_dict["zomuon_perturbed_forward_evals"] = num_directions
        log_dict["zomuon_total_forward_evals"] = self.num_queries
        log_dict["zomuon_projected_scalar_mean"] = projected_scalar_mean
        log_dict[
            "zomuon_projected_scalar_abs_mean"
        ] = projected_scalar_abs_mean
        log_dict["zomuon_perturbed_loss_mean"] = float(
            np.mean(perturbed_losses)
        )
        log_dict["zomuon_basis_refreshed"] = int(basis_refreshed)
        log_dict["zomuon_refresh_count"] = self.basis_refresh_count
        log_dict[
            "zomuon_basis_refresh_time_sec"
        ] = basis_refresh_time
        log_dict["zomuon_rank_requested"] = self.rank
        log_dict[
            "zomuon_effective_rank_min"
        ] = self._effective_rank_min
        log_dict[
            "zomuon_effective_rank_max"
        ] = self._effective_rank_max
        log_dict["zomuon_effective_dim"] = self._effective_dim
        log_dict["zomuon_basis_memory_gb"] = self._basis_memory_gb
        log_dict[
            "zomuon_matrix_parameter_count"
        ] = self._matrix_parameter_count
        log_dict[
            "zomuon_nonmatrix_parameter_count"
        ] = self._nonmatrix_parameter_count
        log_dict["zomuon_matrix_lr"] = self.state.learning_rate
        log_dict["zomuon_nonmatrix_lr"] = self.zomuon_nonmatrix_lr
