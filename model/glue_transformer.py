import math
import os
import pickle
from typing import Optional

import lightning as L
import numpy as np
import torch
from common.logging import logger as logging
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          OPTForSequenceClassification, LlamaForSequenceClassification)

from .prompt_finetune import RobertaModelForPromptFinetuning
from .roberta import RobertaConfig

MODEL_NAME = "model.pt"
OPTIMIZER_NAME = "optimizer.pt"
DATA_MODULE_NAME = "data_module.pkl"
TRAIN_STATE_NAME = "train_state.pkl"
WANDB_INFO_NAME = "wandb_info.pkl"
STEP_INFO_NAME = "step_info.pkl"


class TrainState:
    def __init__(self):
        self.validation_step_outputs = []
        self.tr_loss = []
        self.tr_loss_minibatch = []
        self.time = []
        self.query = []
        self.grad_norm = []
        self.proj_val = []
        self.z_grad = []
        self.val_loss_ls = []
        self.val_acc = []
        self.lr_list = []
        self.memory_usage = []
        self.global_training_steps = 0
        self.learning_rate = 0
        self.learning_rate_aux = 0
        self.full_grad = None


class GLUETransformer(L.LightningModule):
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
        num_queries: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model_name = model_name_or_path
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.soft_prompt = soft_prompt
        self.lora = lora

        if model_init_seed is not None:
            original_seed = torch.initial_seed()
            torch.manual_seed(model_init_seed)

        if model_name_or_path == 'distilbert-base-cased' or model_name_or_path == 'roberta-large':
            if self.soft_prompt is True and model_name_or_path == 'roberta-large':
                config = RobertaConfig.from_pretrained(
                    'roberta-large',
                    num_labels=num_labels,
                    finetuning_task=self.hparams.task_name)
                self.model = RobertaModelForPromptFinetuning.from_pretrained(
                    "roberta-large",
                    config=config
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        elif 'gpt2' in model_name_or_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
            self.model.config.pad_token_id = self.model.config.eos_token_id
        elif 'opt' in model_name_or_path:
            self.model = OPTForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
        elif 'llama' in model_name_or_path or self.config.model_type == 'llama':
            self.model = LlamaForSequenceClassification.from_pretrained(model_name_or_path, config = self.config,
                                                                        use_auth_token=True)
            self.model.config.pad_token_id = self.model.config.eos_token_id
        else:
            raise NotImplementedError(f"Model {model_name_or_path} not supported yet.")

        if model_init_seed is not None:
            torch.manual_seed(original_seed)

        logging.debug(self.model)

        self.use_SGD = use_SGD
        self.zero_order_eps = zero_order_eps
        self.minibatch = minibatch
        self.max_norm = max_norm
        self.z_std = z_std
        self.lr_anneal = lr_anneal
        self.full_parameter = full_parameter
        self.batchsize_limit = batchsize_limit
        self.logger_type = logger_type
        self.num_queries = num_queries

        # train state
        self.state = TrainState()
        self.state.validation_step_outputs = []
        self.state.tr_loss = []
        self.state.tr_loss_minibatch = []
        self.state.time = []
        self.state.query = []
        self.state.grad_norm = []
        self.state.proj_val = []
        self.state.z_grad = []
        self.state.val_loss_ls = []
        self.state.val_acc = []
        self.state.lr_list = []
        self.state.memory_usage = []

        self.state.global_training_steps = 0
        self.state.learning_rate = self.hparams.learning_rate
        self.state.learning_rate_aux = learning_rate_aux

    def forward(self, **inputs):
        return self.model(**inputs)

    def forward_ZO_val(self, inputs):
        model = self.model
        model.eval()
        batch_size = inputs['input_ids'].shape[0]
        iterations = math.ceil(batch_size/self.batchsize_limit)

        loss_sum = 0
        correct_sum = 0
        total_samples = 0

        for i in range(iterations):
            input_batch = {}
            for k, v in inputs.items():
                input_batch[k] = v[i*self.batchsize_limit:min((i+1)*self.batchsize_limit, batch_size)]

            with torch.no_grad():
                outputs = model(**input_batch)

            logits = outputs[1]
            if self.hparams.num_labels > 1:
                preds = torch.argmax(logits, axis=1)
            elif self.hparams.num_labels == 1:
                preds = logits.squeeze()

            labels = input_batch["labels"]
            correct_sum += (preds == labels).sum().item()
            total_samples += len(labels)

            loss_sum += outputs[0].float() * len(labels)  # loss is averaged over samples

            torch.cuda.empty_cache()

        # freeing up memory
        del input_batch, outputs, logits, preds
        return loss_sum, correct_sum, total_samples

    def configure_params(self):
        model = self.model
        if self.full_parameter:
            self.params = [(n, p) for n, p in model.named_parameters()]
        else:
            raise NotImplementedError("partial optimization not supported yet.")

    def configure_optimizers(self):
        """Prepare optimizer"""
        model = self.model

        if self.full_parameter:
            self.params_to_opt = model.parameters()
        else:
            raise NotImplementedError("partial optimization not supported yet.")

        if self.use_SGD:
            optimizer = torch.optim.SGD(self.params_to_opt, lr=self.hparams.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.params_to_opt, lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        loss = outputs['loss']
        self.state.tr_loss.append(loss.detach().cpu().float().numpy())
        logging.info(f'train_loss : {loss.item():.4f}')
        self.log('train_loss', loss.item())
        self.measure_memory_usage()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        self.state.val_loss_ls.append(val_loss.detach().cpu().float().numpy())

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        # Compute Validation Accuracy
        correct_predictions = (preds == labels).sum().item()
        total_samples = len(labels)
        accuracy = correct_predictions / total_samples
        self.state.val_acc.append(accuracy)

        val_info = {"loss": val_loss, "preds": preds, "labels": labels}
        self.state.validation_step_outputs.append(val_info)

        return val_info

    def on_validation_epoch_end(self):
        if len(self.state.validation_step_outputs) == 0:
            return

        loss_sum = torch.stack(
            [x['loss'] * len(x['labels']) for x in self.state.validation_step_outputs]
        ).sum()
        val_preds = torch.cat([x['preds'] for x in self.state.validation_step_outputs])
        val_labels = torch.cat([x['labels'] for x in self.state.validation_step_outputs])

        avg_loss = loss_sum.item() / len(val_labels)
        val_acc = (val_preds == val_labels).sum().item() / len(val_labels)
        logging.info(f'val_loss: {avg_loss:.4f}, val_acc: {val_acc:.3f}')

        self.log('val_loss', avg_loss)
        self.log('val_acc', val_acc)

        self.state.validation_step_outputs.clear()
        return {"val_loss": avg_loss, "val_acc": val_acc}

    @torch.no_grad()
    def zo_forward_memory_eff(self, model, inputs):
        model.eval()
        batch_size = inputs['input_ids'].shape[0]
        iterations = math.ceil(batch_size/self.batchsize_limit)
        loss = 0.0

        for i in range(iterations):
            input_batch = {}
            for k, v in inputs.items():
                input_batch[k] = v[i*self.batchsize_limit:min((i+1)*self.batchsize_limit, batch_size)]

            with torch.inference_mode():
                outputs = model(**input_batch)

            loss += outputs[0].float()
        return loss / iterations

    def zo_forward(self, model, inputs):
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        return loss.detach()

    def efficient_perturb_parameters(self,
                                     parameters,
                                     random_seed: int,
                                     uniform: bool = False,
                                     use_beta: bool = False,
                                     scaling_factor=1):
        torch.manual_seed(random_seed)
        e = self.beta if use_beta else self.zero_order_eps
        for _, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.randn(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=self.z_std, size=param.data.size(),
                                 device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * e
        return

    def efficient_perturb_parameters_layerwise(self,
                                               parameters,
                                               seed_list: list,
                                               uniform: bool = False,
                                               use_beta: bool = False,
                                               scaling_factor=1):
        # assert the same size of seed list and trainable parameters
        assert len(seed_list) == len(parameters)
        e = self.beta if use_beta else self.zero_order_eps
        for (_, param), seed in zip(parameters, seed_list):
            torch.manual_seed(seed)
            if uniform:
                # uniform distribution over unit sphere
                z = torch.randn(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=self.z_std, size=param.data.size(),
                                 device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * e
        return

    def log_training_loss(self, model, batch, fullbatch=False, tb_writer=None):
        with torch.no_grad():
            # loss computation
            loss = self.zo_forward_memory_eff(model, batch)
        if fullbatch:
            logging.info(f"Fullbatch Train Loss : {loss:.3f}")
            self.state.tr_loss.append(loss.detach().cpu().float().numpy())
            if tb_writer is not None:
                tb_writer.add_scalar("fb_train_loss", loss.item(), self.state.global_training_steps)
        else:
            logging.info(f"Minibatch Train Loss : {loss:.3f}")
            self.state.tr_loss_minibatch.append(loss.detach().cpu().float().numpy())
            if tb_writer is not None:
                tb_writer.add_scalar("mb_train_loss", loss.item(), self.state.global_training_steps)

    def forward_difference_grad_est(self, model, parameters, batch, uniform=False):
        random_seed = np.random.randint(1000000000, size=1)
        with torch.no_grad():
            # first function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
            loss1 = self.zo_forward_memory_eff(model, batch)
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform, scaling_factor=-1)
            loss2 = self.zo_forward_memory_eff(model, batch)
        proj_grad = (loss1 - loss2)/self.zero_order_eps
        self.state.proj_val.append(torch.abs(proj_grad).detach().cpu().float().numpy())

        estimator = {}
        torch.manual_seed(random_seed)
        for name, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.randn(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=self.z_std, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            estimator[name] = proj_grad*z
        return estimator

    def clip_gradients_dict(self, grad_dict):
        """Clip the gradients in a dictionary to a maximum norm."""
        total_norm = 0
        for param_name, grad in grad_dict.items():
            total_norm += grad.norm(2) ** 2
        total_norm = total_norm ** 0.5
        logging.debug(f'Norm: {total_norm}')
        self.state.grad_norm.append(total_norm)
        clip_coef = self.state.max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for param_name, grad in grad_dict.items():
                grad_dict[param_name].mul_(clip_coef)

        return grad_dict

    def compute_grad_norm_zo(self, grad_dict):
        """Compute gradient norm."""
        total_norm = 0
        for param_name, grad in grad_dict.items():
            total_norm += grad.norm(2) ** 2
        total_norm = total_norm ** 0.5
        logging.debug(f'Gradient Norm: {total_norm}')
        self.state.grad_norm.append(total_norm)

    def compute_grad_norm_fo(self, param):
        total_norm = 0
        for p in param:
            if not p.detach().grad is None:
                param_norm = p.detach().grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        logging.debug(f'Grad Norm: {total_norm}')
        self.state.grad_norm.append(total_norm)

    def get_average_tensors(self, tensor_list, n=50):
        v1 = torch.stack(tensor_list[-n:]).mean(dim=0)
        v2 = torch.stack(tensor_list[-2*n:-n]).mean(dim=0)
        return v1, v2

    def get_average_np(self, np_list, n=50):
        ll = np.array(np_list)
        v1 = np.mean(ll[-n:])
        v2 = np.mean(ll[-3*n:-2*n])
        return v1, v2

    def measure_memory_usage(self):
        device = next(self.model.parameters()).device
        allocated_memory_bytes = torch.cuda.memory_reserved(device)
        allocated_memory_gb = allocated_memory_bytes / (1024 ** 3)
        self.state.memory_usage.append(allocated_memory_gb)
        # logging.info(f'Memory usage (GB): {allocated_memory_gb:.2f}')
        return allocated_memory_gb

    def load_from_checkpoint(self, checkpoint_path: str):
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, MODEL_NAME)))
        self.state = pickle.load(open(os.path.join(checkpoint_path, TRAIN_STATE_NAME), 'rb'))

    def save_checkpoint(self, checkpoint_path: str):
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(checkpoint_path, MODEL_NAME))
        pickle.dump(self.state, open(os.path.join(checkpoint_path, TRAIN_STATE_NAME), 'wb'))
