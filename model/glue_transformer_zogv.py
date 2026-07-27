import math
import time
from typing import Optional

import numpy as np
import torch

from .glue_transformer_mezo import GLUETransformerMeZo


class GLUETransformerZOGV(GLUETransformerMeZo):
    """Standard MeZO-GV/ZO-GV with ENGINE's existing ZO interface."""

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
        weight_decay: float = 0.01,
        num_queries: int = 4,
        gv_split_ratio: float = 0.5,
        **kwargs,
    ):
        if not full_parameter:
            raise NotImplementedError(
                "ZO-GV currently supports full-parameter optimization only."
            )
        if zero_order_eps <= 0:
            raise ValueError("ZO-GV perturbation_scale must be positive.")
        if num_queries < 4:
            raise ValueError("ZO-GV num_queries must be at least 4.")
        if weight_decay < 0:
            raise ValueError("ZO-GV weight_decay must be non-negative.")
        if not math.isfinite(gv_split_ratio) or not 0 < gv_split_ratio <= 0.5:
            raise ValueError("ZO-GV gv_split_ratio must be in (0, 0.5].")

        candidate_count = int(num_queries) - 2
        elite_count = math.floor(float(gv_split_ratio) * candidate_count)
        if elite_count < 1:
            raise ValueError(
                "ZO-GV gv_split_ratio selects no candidates; increase "
                "gv_split_ratio or num_queries."
            )

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
            weight_decay=weight_decay,
            num_queries=num_queries,
            gv_split_ratio=gv_split_ratio,
            **kwargs,
        )

        self.weight_decay = float(weight_decay)
        self.gv_split_ratio = float(gv_split_ratio)
        self.candidate_count = candidate_count
        self.elite_count = elite_count
        self.state.num_forward_passes = 0

    def configure_params(self):
        super().configure_params()
        self.params = [
            (name, param)
            for name, param in self.params
            if param.requires_grad
        ]
        if not self.params:
            raise ValueError("ZO-GV found no trainable parameters.")
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

    def _make_candidate_direction(
        self,
        param: torch.nn.Parameter,
        param_index: int,
        candidate_seed: int,
    ):
        generator = self._make_generator(
            param.device,
            candidate_seed,
            param_index,
            stream=0,
        )
        direction = torch.randn(
            param.shape,
            device=param.device,
            dtype=param.dtype,
            generator=generator,
        )
        direction.mul_(self.z_std)
        return direction

    def _make_guided_direction(
        self,
        param: torch.nn.Parameter,
        param_index: int,
        low_seeds,
        high_seeds,
    ):
        if not low_seeds or len(low_seeds) != len(high_seeds):
            raise ValueError(
                "ZO-GV low/high candidate groups must be non-empty and equal-sized."
            )

        direction = torch.zeros_like(param)
        group_scale = 1.0 / len(low_seeds)
        for seed in low_seeds:
            candidate = self._make_candidate_direction(
                param,
                param_index,
                seed,
            )
            direction.add_(candidate, alpha=group_scale)
        for seed in high_seeds:
            candidate = self._make_candidate_direction(
                param,
                param_index,
                seed,
            )
            direction.add_(candidate, alpha=-group_scale)
        return direction

    @torch.no_grad()
    def _perturb_candidate(
        self,
        candidate_seed: int,
        scaling_factor: float,
    ):
        alpha = scaling_factor * self.zero_order_eps
        for param_index, (_, param) in enumerate(self.params):
            direction = self._make_candidate_direction(
                param,
                param_index,
                candidate_seed,
            )
            param.add_(direction, alpha=alpha)

    @staticmethod
    def _finite_loss_value(loss, context: str) -> float:
        value = float(loss.item())
        if not math.isfinite(value):
            raise FloatingPointError(
                f"ZO-GV encountered a non-finite {context}: {value}."
            )
        return value

    @torch.no_grad()
    def _score_candidate(self, model, batch, candidate_seed: int) -> float:
        position = 0
        try:
            self._perturb_candidate(candidate_seed, scaling_factor=1.0)
            position = 1
            loss = self.zo_forward_memory_eff(model, batch)
            return self._finite_loss_value(loss, "candidate loss")
        finally:
            if position == 1:
                self._perturb_candidate(candidate_seed, scaling_factor=-1.0)

    def _select_guiding_groups(self, candidate_seeds, candidate_losses):
        if len(candidate_seeds) != self.candidate_count:
            raise ValueError(
                "ZO-GV candidate seed count does not match num_queries - 2."
            )
        if len(candidate_losses) != len(candidate_seeds):
            raise ValueError(
                "ZO-GV candidate seed and loss counts must match."
            )
        if not all(math.isfinite(float(loss)) for loss in candidate_losses):
            raise FloatingPointError("ZO-GV candidate losses must all be finite.")

        order = np.argsort(np.asarray(candidate_losses), kind="stable")
        low_indices = order[:self.elite_count]
        high_indices = order[-self.elite_count:]
        low_seeds = [candidate_seeds[int(index)] for index in low_indices]
        high_seeds = [candidate_seeds[int(index)] for index in high_indices]
        return low_seeds, high_seeds

    @torch.no_grad()
    def _perturb_guided(
        self,
        low_seeds,
        high_seeds,
        scaling_factor: float,
    ):
        alpha = scaling_factor * self.zero_order_eps
        for param_index, (_, param) in enumerate(self.params):
            direction = self._make_guided_direction(
                param,
                param_index,
                low_seeds,
                high_seeds,
            )
            param.add_(direction, alpha=alpha)

    @torch.no_grad()
    def _guided_central_difference(
        self,
        model,
        batch,
        low_seeds,
        high_seeds,
    ):
        position = 0
        try:
            self._perturb_guided(
                low_seeds,
                high_seeds,
                scaling_factor=1.0,
            )
            position = 1
            loss_plus = self.zo_forward_memory_eff(model, batch)

            self._perturb_guided(
                low_seeds,
                high_seeds,
                scaling_factor=-2.0,
            )
            position = -1
            loss_minus = self.zo_forward_memory_eff(model, batch)
        finally:
            if position == 1:
                self._perturb_guided(
                    low_seeds,
                    high_seeds,
                    scaling_factor=-1.0,
                )
            elif position == -1:
                self._perturb_guided(
                    low_seeds,
                    high_seeds,
                    scaling_factor=1.0,
                )

        loss_plus_value = self._finite_loss_value(loss_plus, "guided L+")
        loss_minus_value = self._finite_loss_value(loss_minus, "guided L-")
        projected_grad = (
            loss_plus_value - loss_minus_value
        ) / (2.0 * self.zero_order_eps)
        if not math.isfinite(projected_grad):
            raise FloatingPointError(
                "ZO-GV encountered a non-finite projected gradient."
            )
        return loss_plus_value, loss_minus_value, projected_grad

    @torch.no_grad()
    def _apply_zogv_update(
        self,
        low_seeds,
        high_seeds,
        projected_grad: float,
    ):
        learning_rate = self.state.learning_rate
        if self.weight_decay:
            decay = 1.0 - learning_rate * self.weight_decay
            for _, param in self.params:
                param.mul_(decay)

        for param_index, (_, param) in enumerate(self.params):
            direction = self._make_guided_direction(
                param,
                param_index,
                low_seeds,
                high_seeds,
            )
            param.add_(
                direction,
                alpha=-learning_rate * projected_grad,
            )

    def _sample_candidate_seeds(self):
        seeds = []
        seen = set()
        while len(seeds) < self.candidate_count:
            seed = int(np.random.randint(1_000_000_000))
            if seed not in seen:
                seen.add(seed)
                seeds.append(seed)
        return seeds

    @torch.no_grad()
    def training_step(self, model, batch, tb_writer=None, **kwargs):
        del tb_writer
        start_time = time.perf_counter()
        log_dict = kwargs["log"]

        candidate_seeds = self._sample_candidate_seeds()
        candidate_losses = [
            self._score_candidate(model, batch, seed)
            for seed in candidate_seeds
        ]
        low_seeds, high_seeds = self._select_guiding_groups(
            candidate_seeds,
            candidate_losses,
        )
        loss_plus, _, projected_grad = self._guided_central_difference(
            model,
            batch,
            low_seeds,
            high_seeds,
        )
        self._apply_zogv_update(
            low_seeds,
            high_seeds,
            projected_grad,
        )

        self.state.global_training_steps += 1
        batch_size = int(batch["input_ids"].shape[0])
        self.state.query.append(batch_size * self.num_queries)
        self.state.num_forward_passes += self.num_queries
        total_time = time.perf_counter() - start_time
        self.state.time.append(total_time)
        self.state.proj_val.append(abs(projected_grad))

        log_dict["train_loss"] = loss_plus
        log_dict["time"] = total_time
        log_dict["num_forward_passes"] = self.state.num_forward_passes
        log_dict["zogv_candidate_count"] = self.candidate_count
        log_dict["zogv_elite_count"] = self.elite_count
        log_dict["zogv_split_ratio"] = self.gv_split_ratio
        log_dict["zogv_candidate_loss_min"] = float(min(candidate_losses))
        log_dict["zogv_candidate_loss_max"] = float(max(candidate_losses))
        log_dict["zogv_candidate_loss_mean"] = float(
            np.mean(candidate_losses)
        )
        log_dict["zogv_projected_grad"] = projected_grad
        log_dict["zogv_projected_grad_abs"] = abs(projected_grad)
        log_dict["zogv_candidate_forward_evals"] = self.candidate_count
        log_dict["zogv_guided_forward_evals"] = 2
        log_dict["zogv_total_forward_evals"] = self.num_queries
