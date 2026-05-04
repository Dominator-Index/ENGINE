import time
import torch
from .glue_transformer import GLUETransformer


class GLUETransformerFO(GLUETransformer):
    def __init__(self, *args, use_sgd=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sgd = use_sgd

    def training_step(self, model, batch, tb_writer=None, **kwargs):
        start_time = time.perf_counter()
        optimizer = kwargs['optimizer']
        log_dict = kwargs['log']

        outputs = model(**batch)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.state.global_training_steps += 1

        # Logging
        total_time = time.perf_counter() - start_time
        n = batch['input_ids'].size(0)
        self.state.time.append(total_time)
        self.state.query.append(n)

        log_dict['train_loss'] = loss.item()
        log_dict['time'] = total_time