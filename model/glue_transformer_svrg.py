import logging
import time
from typing import Optional

import numpy as np
import torch

from .glue_transformer_mezo import GLUETransformerMeZo


class GLUETransformerMeZoSVRG(GLUETransformerMeZo):
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
        q: int = 1,
        **kwargs,
    ):
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
            model_init_seed)

        self.q = q

    def training_step(self, model, batch, tb_writer=None, **kwargs):
        # run MeZO-SVRG update
        n = batch['input_ids'].size(0)
        logging.debug(f'Batch size: {n}')
        start_time = time.perf_counter()
        log_dict = kwargs['log']

        # current iteration
        curr_iter = kwargs['curr_iter']
        total_batches = kwargs['full_batch_size']
        w = 2 * total_batches
        full_batch = kwargs['is_full_batch']

        # learning rate scheduling strategy
        if len(self.state.tr_loss_minibatch) > 2*w and curr_iter % total_batches == 0:
            v1, v2 = self.get_average_np(self.state.tr_loss_minibatch, int(w/2))
            logging.debug(f'leading average: {v1}')
            logging.debug(f'trailing average: {v2}')
            if v1/v2 > 1.05:
                self.state.learning_rate_aux = max(self.state.learning_rate_aux/self.lr_anneal, 1e-5)
                self.state.learning_rate = max(self.state.learning_rate/self.lr_anneal, 1e-6)
        logging.debug(f'Learning rate (full-batch): {self.state.learning_rate_aux}')
        logging.debug(f'Learning rate (mini-batch): {self.state.learning_rate}')
        self.state.lr_list.append(self.state.learning_rate_aux)

        # parameters contains tuples of params to optimize in list
        parameters = self.params

        # do full batch update every q steps
        if full_batch is True:
            self.state.full_grad = self.SPSA_estimator(model, parameters, batch)
            self.parameters = parameters.copy()
            with torch.no_grad():
                for name, param in parameters:
                    param.data = param.data - self.state.learning_rate_aux * self.state.full_grad[name]
        else:
            # minibatch update
            parameters = self.SPSA_estimator_me(model, parameters, batch, scale=-1)  # in-place operation
            parameters = self.SPSA_estimator_me(model, self.parameters, batch)  # in-place operation

            with torch.no_grad():
                for name, param in parameters:
                    param.data = param.data - self.state.learning_rate * self.state.full_grad[name]

        self.state.global_training_steps += 1

        total_time = time.perf_counter() - start_time
        self.state.time.append(total_time)
        self.state.query.append(2*n)

        with torch.no_grad():
            # loss computation
            loss = self.zo_forward_memory_eff(model, batch)

        # self.log_training_loss(model, batch,
        #                        fullbatch=True if curr_iter % self.q == 0 else False,
        #                        tb_writer=tb_writer)
        if full_batch:
            self.state.tr_loss.append(loss.detach().cpu().float().numpy())
            log_dict['fb_train_loss'] = loss.item()
        else:
            self.state.tr_loss_minibatch.append(loss.detach().cpu().float().numpy())
            log_dict['train_loss'] = loss.item()

        log_dict['time'] = total_time
        # log_dict['memory'] = self.measure_memory_usage()

    def SPSA_estimator(self, model, parameters, batch):
        return self.central_difference_grad_est(model, parameters, batch)

    def SPSA_estimator_me(self, model, parameters, batch, scale=1):
        # memory efficient SPSA estimator
        return self.central_difference_grad_est_me(model, parameters, batch, scale)

    def central_difference_grad_est(self, model, parameters, batch, uniform=False):
        random_seed = np.random.randint(1000000000, size=1)
        with torch.no_grad():
            # first function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
            loss1 = self.zo_forward_memory_eff(model, batch)
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform, scaling_factor=-2)
            loss2 = self.zo_forward_memory_eff(model, batch)
        proj_grad = (loss1 - loss2)/(2 * self.zero_order_eps)
        model_dtype = next(self.model.parameters()).dtype
        proj_grad = proj_grad.to(model_dtype)
        logging.debug(f'Projected Grad: {proj_grad}')
        self.state.proj_val.append(torch.abs(proj_grad).detach().cpu().float().numpy())

        estimator = {}
        self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
        torch.manual_seed(random_seed)
        for name, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.randn(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=self.z_std, size=param.data.size(),
                                 device=param.data.device, dtype=param.data.dtype)
            estimator[name] = proj_grad*z
        return estimator

    def central_difference_grad_est_me(self, model, parameters, batch, scale, uniform=False):
        # memory efficient central difference spsa estimator
        random_seed = np.random.randint(1000000000, size=1)
        with torch.no_grad():
            # first function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
            loss1 = self.zo_forward_memory_eff(model, batch)
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform, scaling_factor=-2)
            loss2 = self.zo_forward_memory_eff(model, batch)
        proj_grad = (loss1 - loss2)/(2 * self.zero_order_eps)
        self.state.proj_val.append(torch.abs(proj_grad).detach().cpu().float().numpy())

        self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
        torch.manual_seed(random_seed)
        for name, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.randn(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=self.z_std, size=param.data.size(),
                                 device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scale * self.state.learning_rate * proj_grad * z
        return parameters
