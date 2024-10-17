from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Mol2MSDataset(Dataset):
    def __init__(
        self, hf_dataset: Dataset, model_name: str, max_length: int
    ):
        self.dataset = hf_dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "</s>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        tokenized_smiles = self.tokenizer(
            item["smiles"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_attention_mask=True,
        )

        return {
            "tokenized_smiles": tokenized_smiles.input_ids.squeeze(),
            "attention_mask": tokenized_smiles.attention_mask.squeeze(),
            "index": item["index"],
            "collision_energy": item["collision_energy"],
            "instrument_type": item["instrument_type"],
            "stop_token": item["stop_token"],
            "mz": item["mzs"],
            "intensity": item["intensities"],
        }
