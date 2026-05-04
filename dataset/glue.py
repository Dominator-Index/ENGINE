from typing import Optional

import datasets
import lightning as L
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from dataset.mezo_dataset import tokenize_multipart_input


GLUE_TASKS = ["cola", "sst2", "mnli", "qnli"]


class GLUEDataModule(L.LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mnli": 3,
        "qnli": 2,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels"
    ]

    glue_task_label_mapping = {
        'sst2': {
            0: 'terrible',
            1: 'great'
        }
    }

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        sample_size: int = 128,
        validation_sample_size: int = 128,
        soft_prompt: bool = False,
        hf_token: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.sample_size = sample_size
        self.validation_sample_size = validation_sample_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, token=hf_token)
        self.soft_prompt = soft_prompt
        if soft_prompt is True:
            self.loader_columns.append("mask_pos")

    def setup(self, stage: str = 'fit', randomize=False, seed=None):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            if split == "test":
                continue

            if self.soft_prompt is True:
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features_soft_prompt,
                    batched=False,
                    remove_columns=["label"],
                    load_from_cache_file=True
                )
            else:
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    remove_columns=["label"],
                    load_from_cache_file=True
                )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

        train_size = len(self.dataset["train"])
        self.subset_indices = list(range(min(self.sample_size, train_size)))
        
        if len(self.eval_splits) == 1:
            val_size = len(self.dataset["validation"])
            self.subset_indices_val = list(range(min(self.validation_sample_size, val_size)))
        else:
            val_size = len(self.dataset["validation_matched"])
            self.subset_indices_val = list(range(min(self.validation_sample_size, val_size)))

        self.subset_train_dataset = Subset(self.dataset["train"], self.subset_indices)
        if len(self.eval_splits) == 1:
            self.subset_val_dataset = Subset(self.dataset["validation"], self.subset_indices_val)
        else:
            self.subset_val_dataset = Subset(self.dataset["validation_matched"], self.subset_indices_val)

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.subset_train_dataset, batch_size=self.train_batch_size, shuffle=True, drop_last=True)

    def train_full_dataloader(self):
        return DataLoader(self.subset_train_dataset, batch_size=self.sample_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.subset_val_dataset, batch_size=self.eval_batch_size, drop_last=True)

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features_soft_prompt(self, example_batch, indices=None):
        if self.task_name == 'sst2':
            inputs = tokenize_multipart_input(
                input_text_list=[example_batch[self.text_fields[0]]],
                max_length=self.max_seq_length,
                tokenizer=self.tokenizer,
                task_name=self.task_name,
                prompt=True,
                template='*cls**sent_0*_It_was*mask*.*sep+*',
                label_word_list={0: ' terrible', 1: ' great'}
            )
            inputs['labels'] = example_batch["label"]
            return inputs
        elif self.task_name == 'mnli':
            inputs = tokenize_multipart_input(
                input_text_list=[example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]],
                max_length=256,
                tokenizer=self.tokenizer,
                task_name=self.task_name,
                prompt=True,
                template='*cls**sent-_0*?*mask*,*+sentl_1**sep+*',
                label_word_list={'contradiction': 'No', 'entailment': 'Yes', 'neutral': 'Maybe'},
                first_sent_limit=240
            )
            inputs['labels'] = example_batch["label"]
            return inputs

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]],
                                           example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        if (self.model_name_or_path == 'distilbert-base-cased' or
            self.model_name_or_path == 'roberta-large'):
            features = self.tokenizer.batch_encode_plus(texts_or_text_pairs,
                                                        max_length=self.max_seq_length,
                                                        padding='max_length',
                                                        truncation=True)
            # Rename label to labels to make it easier to pass to model forward
            features["labels"] = example_batch["label"]
        elif ('gpt2' in self.model_name_or_path or 'opt' in self.model_name_or_path or
              'llama' in self.model_name_or_path.lower()):
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            features = self.tokenizer.batch_encode_plus(texts_or_text_pairs,
                                                        max_length=self.max_seq_length,
                                                        padding='max_length',
                                                        truncation=True)
            features["labels"] = example_batch["label"]

        return features
