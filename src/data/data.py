from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Mol2MSDataset(Dataset):
    def __init__(
        self, hf_dataset: Dataset, model_name: str, max_ms_length: int, max_length: int
    ):
        self.dataset = hf_dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_ms_length = max_ms_length
        self.max_length = max_length

        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "</s>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        smiles_encoding = self.tokenizer(
            item["smiles"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_attention_mask=True,
        )

        ms_data: List[Tuple[float, float, int]] = []
        for i, (mz, intensity) in enumerate(zip(item["mzs"], item["intensities"])):
            ms_data.append((round(mz, 4), round(intensity, 4), i))

        ms_data.sort(key=lambda x: x[0])
        ms_data = ms_data[: self.max_ms_length]  # Truncate if too long

        mzs, intensities, indices = zip(*ms_data) if ms_data else ([], [], [])

        mz_tensor = torch.tensor(mzs, dtype=torch.float32)
        intensity_tensor = torch.tensor(intensities, dtype=torch.float32)
        index_tensor = torch.tensor(indices, dtype=torch.long)

        create_next_token_tensor = torch.ones(self.max_ms_length, dtype=torch.long)
        if len(mzs) < self.max_ms_length:
            create_next_token_tensor[len(mzs) - 1 :] = 0
        else:
            create_next_token_tensor[-1] = 0

        # pad if necessary
        if len(mz_tensor) < self.max_ms_length:
            padding_length = self.max_ms_length - len(mz_tensor)
            mz_tensor = torch.nn.functional.pad(mz_tensor, (0, padding_length))
            intensity_tensor = torch.nn.functional.pad(
                intensity_tensor, (0, padding_length)
            )
            index_tensor = torch.nn.functional.pad(
                index_tensor, (0, padding_length), value=self.max_ms_length - 1
            )

        return {
            "input_ids": smiles_encoding.input_ids.squeeze(),
            "attention_mask": smiles_encoding.attention_mask.squeeze(),
            "mz": mz_tensor,
            "intensity": intensity_tensor,
            "index": index_tensor,
            "create_next_token": create_next_token_tensor,
        }
