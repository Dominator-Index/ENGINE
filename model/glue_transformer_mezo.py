import time

import numpy as np
import torch

from .glue_transformer import GLUETransformer


class GLUETransformerMeZo(GLUETransformer):
    def perturb_parameters(self, random_seed: int, scaling_factor=1):
        torch.manual_seed(random_seed)

        for name, param in self.params:
            z = torch.normal(mean=0, std=self.z_std, size=param.data.size(),
                             device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.zero_order_eps

    def zo_step(self, model, batch, random_seed):
        """
        Zeroth-order gradient estimation
        """
        # first function evaluation
        self.perturb_parameters(random_seed, scaling_factor=1)
        loss1 = self.zo_forward_memory_eff(model, batch)

        # second function evaluation
        self.perturb_parameters(random_seed, scaling_factor=-2)
        loss2 = self.zo_forward_memory_eff(model, batch)

        projected_grad = ((loss1 - loss2) / (2 * self.zero_order_eps)).item()

        # model_dtype = next(self.model.parameters()).dtype
        # projected_grad = projected_grad.to(model_dtype)

        # reset model back to its parameters at start of step
        self.perturb_parameters(random_seed, scaling_factor=1)

        return loss1, projected_grad

    @torch.no_grad()
    def zo_update(self, random_seed, projected_grad, scale=-1):
        torch.manual_seed(random_seed)

        # compute SPSA gradient estimator
        for _, param in self.params:
            u = torch.normal(mean=0, std=1, size=param.data.size(),
                             device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scale * self.state.learning_rate * projected_grad * u

    def training_step(self, model, batch, tb_writer=None, **kwargs):
        # run ZO update
        num_grad_est = self.num_queries // 2
        n = batch['input_ids'].size(0)
        start_time = time.perf_counter()
        random_seeds = [np.random.randint(1000000000) for _ in range(num_grad_est)]
        log_dict = kwargs['log']

        losses = []
        for k in range(num_grad_est):
            loss, projected_grad = self.zo_step(model, batch, random_seeds[k])
            losses.append(loss.item())
            mean_proj_grad = projected_grad / num_grad_est
            self.zo_update(random_seeds[k], mean_proj_grad)

        lmean = np.mean(losses)
        self.state.global_training_steps += 1

        # logging
        total_time = time.perf_counter() - start_time

        self.state.time.append(total_time)
        self.state.query.append(2*n*num_grad_est)

        log_dict['train_loss'] = lmean.item()
        log_dict['time'] = total_time
