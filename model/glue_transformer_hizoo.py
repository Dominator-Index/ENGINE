import time
from typing import Optional

import numpy as np
import torch

from .glue_transformer_mezo import GLUETransformerMeZo
from common.hessian_smooth_scheduler import Hessian_smooth_scheduler

class GLUETransformerHiZOO(GLUETransformerMeZo):
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
        num_queries: int = 10,
        weight_decay: float = 0.01,
        hessian_smooth_type: str = "constant1e-8",
        num_epochs: int = 3000,
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
        self.Hessian_matrix = {}
        self.hessian_smooth_type = hessian_smooth_type
        self.num_epochs = num_epochs
        self.num_queries = num_queries
        self.weight_decay = weight_decay

    def configure_params(self):
        super().configure_params()

        for name, param in self.params:
            if param.requires_grad:
                self.Hessian_matrix[name] = torch.ones(size=param.data.size(), device=param.data.device,
                                                       dtype=param.data.dtype)

    @torch.no_grad()
    def efficient_Hessian_perturb_parameters(self, random_seed: int, Hessian_matrix=None,
                                             scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.params:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor / torch.sqrt(Hessian_matrix[name]) * z * self.zero_order_eps

    def zo_Hessian_step(self, model, batch, random_seed):
        loss_original = self.zo_forward_memory_eff(model, batch)

        self.efficient_Hessian_perturb_parameters(random_seed, self.Hessian_matrix, scaling_factor=1)
        loss1 = self.zo_forward_memory_eff(model, batch)
        self.efficient_Hessian_perturb_parameters(random_seed, self.Hessian_matrix, scaling_factor=-2)
        loss2 = self.zo_forward_memory_eff(model, batch)
        self.efficient_Hessian_perturb_parameters(random_seed, self.Hessian_matrix, scaling_factor=1)
        return loss_original, loss1, loss2

    def zo_Hessian_update(self, random_seed, loss_original, loss1, loss2, Hessian_smooth, num_grad_est):
        torch.manual_seed(random_seed)
        for name, param in self.params:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                             dtype=param.data.dtype)

            Hessian_temp = self.Hessian_matrix[name] * z * z
            Hessian_estimator = (torch.abs(loss1 + loss2 - 2 * loss_original) * Hessian_temp * Hessian_smooth / (
                        2 * self.zero_order_eps * self.zero_order_eps))

            self.Hessian_matrix[name] = ((1 - Hessian_smooth) * self.Hessian_matrix[name] + Hessian_estimator)

            grad = (loss1 - loss2) / (2 * self.zero_order_eps) * z / torch.sqrt(self.Hessian_matrix[name])
            grad = grad / num_grad_est
            param.data = param.data - self.state.learning_rate * (grad + self.weight_decay * param.data)

    def training_step(self, model, batch, tb_writer=None, **kwargs):
        num_grad_est = self.num_queries // 3
        n = batch['input_ids'].size(0)
        start_time = time.perf_counter()
        random_seeds = [np.random.randint(1000000000) for _ in range(num_grad_est)]
        log_dict = kwargs['log']

        Hessian_smooth = Hessian_smooth_scheduler(self.hessian_smooth_type, self.state.global_training_steps,
                                                  self.num_epochs)

        losses = []
        for k in range(num_grad_est):
            loss_original, loss1, loss2 = self.zo_Hessian_step(model, batch, random_seeds[k])
            losses.append(loss_original.item())
            self.zo_Hessian_update(random_seeds[k], loss_original, loss1, loss2, Hessian_smooth, num_grad_est)

        lmean = np.mean(losses)
        self.state.global_training_steps += 1

        # logging
        total_time = time.perf_counter() - start_time

        self.state.time.append(total_time)
        self.state.query.append(3 * n * num_grad_est)

        log_dict['train_loss'] = lmean.item()
        log_dict['time'] = total_time