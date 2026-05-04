import datasets
import numpy as np
import pytorch_lightning as pl
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from common.utils import temp_seed


SUPERGLUE_TASKS = ["rte", "wic", "copa", "boolq", "cb", "wsc"]


class SuperGLUEDataModule(pl.LightningDataModule):
    task_num_labels = {
        "rte": 2,
        "wic": 2,
        "copa": 2,
        "boolq": 2,
        "cb": 3,
        "wsc": 2,
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

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "rte",
        max_seq_length: int = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        sample_size: int = 128,
        validation_sample_size: int = 128,
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

        self.num_labels = self.task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def sample_subset(self, split="train", seed=0, num=100):
        with temp_seed(seed):
            lens = len(self.dataset[split])
            index = np.random.permutation(lens).tolist()[:num + 1]
            index = index[:num]
            return index

    def load_dataset(self, task_name):
        hf_dataset = datasets.load_dataset("super_glue", task_name)
        dataset = DatasetDict()
        if task_name == "rte":
            dataset = hf_dataset.map(lambda x: {"input": (x["premise"], x["hypothesis"])})
        elif task_name == "boolq":
            dataset = hf_dataset.map(lambda x: {"input": (x["passage"], x["question"]+"?" if x["question"][-1] != "?" else x["question"])})
        elif task_name == "copa":
            dataset = DatasetDict()
            for split in hf_dataset.keys():
                examples = []
                for example in hf_dataset[split]:
                    premise = example["premise"]
                    choice1 = example["choice1"]
                    choice2 = example["choice2"]
                    question = example["question"]

                    if question == "cause":
                        examples.append({
                            "input": [choice1, premise],
                            "label": int(example["label"] == 0) if example["label"] is not None else None
                        })
                        examples.append({
                            "input": [choice2, premise],
                            "label": int(example["label"] == 1) if example["label"] is not None else None
                        })
                    else:
                        examples.append({
                            "input": [premise, choice1],
                            "label": int(example["label"] == 0) if example["label"] is not None else None
                        })
                        examples.append({
                            "input": [premise, choice2],
                            "label": int(example["label"] == 1) if example["label"] is not None else None
                        })
                dataset[split] = Dataset.from_list(examples)

        elif task_name == "cb":
            dataset = hf_dataset.map(lambda x: {"input": (x["premise"], x["hypothesis"])})
        elif task_name == "wic":
            dataset = hf_dataset.map(lambda x: {"input": (f"{x['word']} : {x['sentence1']}", x['sentence2'])})
        elif task_name == "wsc":
            def wsc_formatter(example):
                text = example['text']
                span1_index = example['span1_index']
                span2_index = example['span2_index']
                span1_text = example['span1_text']
                span2_text = example['span2_text']

                words = text.split()
                span1_word_len = len(span1_text.split())
                span2_word_len = len(span2_text.split())

                words[span1_index:span1_index + span1_word_len] = [f"# {span1_text} #"]

                if span1_word_len > 1:
                    # If the span is more than one word, we need to adjust the index of the second span since the first more than one word has been replaced by a single word
                    span2_index -= span1_word_len - 1
                words[span2_index:span2_index + span2_word_len] = [f"* {span2_text} *"]

                return " ".join(words)
            dataset = hf_dataset.map(lambda x: {"input": wsc_formatter(x)})

        return dataset

    def setup(self, stage='fit', randomize=False, seed=None):
        self.dataset = self.load_dataset(self.task_name)

        for split in self.dataset.keys():
            if split == "test":
                continue

            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
                load_from_cache_file=True
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        if len(self.dataset["train"]) < self.sample_size:
            self.sample_size = len(self.dataset["train"])

        self.eval_splits = ["validation"]
        if len(self.dataset[self.eval_splits[0]]) < self.validation_sample_size:
            self.validation_sample_size = len(self.dataset[self.eval_splits[0]])

        self.subset_indices = list(range(self.sample_size))
        self.subset_indices_val = list(range(self.validation_sample_size))

        self.subset_train_dataset = Subset(self.dataset["train"], self.subset_indices)
        self.subset_val_dataset = Subset(self.dataset["validation"], self.subset_indices_val)

    def prepare_data(self):
        datasets.load_dataset("super_glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.subset_train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def train_full_dataloader(self):
        return DataLoader(self.subset_train_dataset, batch_size=self.sample_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.subset_val_dataset, batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)

    def convert_to_features(self, example_batch, indices=None):
        if self.tokenizer.pad_token_id is None:
            # GPT2 and OPT
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        features = self.tokenizer.batch_encode_plus(
            example_batch["input"], max_length=self.max_seq_length, padding='max_length', truncation=True
        )
        features["labels"] = example_batch["label"]

        return features
