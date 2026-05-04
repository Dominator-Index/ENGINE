import time
import numpy as np
import torch
from .glue_transformer_mezo import GLUETransformerMeZo


class GLUETransformerMeZoAdam(GLUETransformerMeZo):
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

    def training_step(self, model, batch, tb_writer=None, **kwargs):
        # run ZO update
        num_grad_est = self.num_queries // 2
        n = batch['input_ids'].size(0)
        start_time = time.perf_counter()
        random_seeds = [np.random.randint(1000000000) for _ in range(num_grad_est)]
        optimizer = kwargs['optimizer']
        log_dict = kwargs['log']


        losses = []
        for k in range(num_grad_est):
            loss, projected_grad = self.zo_step(model, batch, random_seeds[k])
            losses.append(loss.item())
            mean_proj_grad = projected_grad / num_grad_est

            # compute SPSA gradient estimator
            torch.manual_seed(random_seeds[k])
            for _, param in self.params:
                param.grad = mean_proj_grad * torch.normal(mean=0, std=1, size=param.data.size(),
                                                           device=param.data.device,
                                                           dtype=param.data.dtype)
                optimizer.step()
                param.grad = None

        lmean = np.mean(losses)
        self.state.global_training_steps += 1

        # logging
        total_time = time.perf_counter() - start_time
        self.state.time.append(total_time)
        self.state.query.append(2*n*num_grad_est)

        log_dict['train_loss'] = lmean.item()
        log_dict['time'] = total_time
