import time
from typing import Optional

import numpy as np
import torch

from .glue_transformer_mezo import GLUETransformerMeZo


class GLUETransformerLoZo(GLUETransformerMeZo):
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
        step_interval: int = 50,  # subspace bases update interval in LoZo
        rank: int = 2,
        weight_decay: float = 0.01,
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
                         model_init_seed)
        self.rank = rank
        self.basis_upd_interval = step_interval
        self.subspace_basis = {}
        self.weight_decay = weight_decay

    @torch.no_grad()
    def lowrank_perturb_parameters(self, random_seed, scaling_factor=1, pos=None):
        torch.manual_seed(random_seed)

        for name, param in self.params:
            if param.data.ndim >= 2:
                V = self.subspace_basis[name]

                U = torch.randn(param.data.size(0), self.rank,
                                device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * (U @ V.t()) * self.zero_order_eps
            else:
                z = torch.normal(mean=0, std=self.z_std, size=param.data.size(),
                                 device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.zero_order_eps

    def lowrank_zo_step(self, model, batch, random_seed, pos=None):
        self.lowrank_perturb_parameters(random_seed, scaling_factor=1, pos=pos)
        loss1 = self.zo_forward_memory_eff(model, batch)

        # second function evaluation
        self.lowrank_perturb_parameters(random_seed, scaling_factor=-2, pos=pos)
        loss2 = self.zo_forward_memory_eff(model, batch)

        projected_grad = ((loss1 - loss2) / (2 * self.zero_order_eps)).item()

        # reset model back to its parameters at start of step
        self.lowrank_perturb_parameters(random_seed, scaling_factor=1, pos=pos)
        return loss1, projected_grad

    @torch.no_grad()
    def _gd_with_decay(self, name, param, grad):
        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            param.mul_(1.0 - self.state.learning_rate * self.weight_decay)

        param.sub_(grad, alpha=self.state.learning_rate)

    def lowrank_zo_update(self, random_seed, proj_grad):
        torch.manual_seed(random_seed)

        for name, param in self.params:
            if param.data.ndim >= 2:
                V = self.subspace_basis[name]
                U = torch.randn(param.data.size(0), self.rank,
                                device=param.data.device, dtype=param.data.dtype)
                grad = proj_grad * (U @ V.t())
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(),
                                 device=param.data.device, dtype=param.data.dtype)
                grad = proj_grad * z

            self._gd_with_decay(name, param, grad)

    def update_basis(self, sub_dim: int = 1):
        for name, param in self.params:
            if param.data.ndim >= 2:
                self.subspace_basis[name] = torch.randn(param.data.size(sub_dim),
                                                        self.rank,
                                                        device=param.data.device,
                                                        dtype=param.data.dtype)

    def training_step(self, model, batch, tb_writer=None, **kwargs):
        num_grad_est = self.num_queries // 2
        n = batch['input_ids'].size(0)
        start_time = time.perf_counter()
        random_seeds = [np.random.randint(1000000000) for _ in range(num_grad_est)]
        log_dict = kwargs['log']

        step = self.state.global_training_steps
        if step % self.basis_upd_interval == 0:
            self.update_basis(sub_dim=1)

        losses = []
        for k in range(num_grad_est):
            loss, proj_grad = self.lowrank_zo_step(model, batch, random_seeds[k])
            losses.append(loss.item())
            mean_proj_grad = proj_grad / num_grad_est
            self.lowrank_zo_update(random_seeds[k], mean_proj_grad)

        lmean = np.mean(losses)
        self.state.global_training_steps += 1

        total_time = time.perf_counter() - start_time

        self.state.time.append(total_time)
        self.state.query.append(2*n*num_grad_est)

        log_dict['train_loss'] = lmean.item()
        log_dict['time'] = total_time
