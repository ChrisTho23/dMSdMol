from typing import Dict, List, Tuple

import torch as t
from jaxtyping import Float, Int
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Mol2MSDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        model_name: str,
        max_encoder_length: int,
        max_decoder_length: int,
    ):
        self.dataset = hf_dataset
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length

        self.mz_pad_value = -100
        self.intensity_pad_value = -100

        self.mz_start_token = 0
        self.intensity_start_token = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize special tokens for the tokenizer."""
        special_tokens = {}
        if self.tokenizer.bos_token is None:
            special_tokens["bos_token"] = "<s>"
        if self.tokenizer.eos_token is None:
            special_tokens["eos_token"] = "</s>"
        if self.tokenizer.pad_token is None:
            special_tokens["pad_token"] = "<pad>"
        if self.tokenizer.cls_token is None:
            special_tokens["cls_token"] = "<cls>"
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)

    def _tokenize(self, smiles_string: str) -> Dict[str, Int[t.Tensor, "1 seq"]]:
        """Tokenizes a SMILES string using the BART tokenizer."""
        tokens = self.tokenizer(
            smiles_string,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_length - 1,  # Reserve space for cls_token
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        cls_token_id = self.tokenizer.cls_token_id
        tokens["input_ids"] = t.cat(
            [t.full((1, 1), cls_token_id), tokens["input_ids"]], dim=1
        )  # 1 seq
        tokens["attention_mask"] = t.cat(
            [t.ones((1, 1), dtype=t.long), tokens["attention_mask"]], dim=1
        )  # 1 seq
        return tokens

    def _pad_ms(
        self, mz: List[float], intensity: List[float]
    ) -> Tuple[Float[t.Tensor, "seq-1"], Float[t.Tensor, "seq-1"]]:
        mz = t.tensor(mz)
        intensity = t.tensor(intensity)

        pad_length = self.max_decoder_length - 1 - len(mz)
        if pad_length > 0:
            mz = t.nn.functional.pad(
                mz, (0, pad_length), mode="constant", value=self.mz_pad_value
            )
            intensity = t.nn.functional.pad(
                intensity,
                (0, pad_length),
                mode="constant",
                value=self.intensity_pad_value,
            )
        else:
            mz = mz[: self.max_decoder_length - 1]
            intensity = intensity[: self.max_decoder_length - 1]
        return mz, intensity

    def _shift_right(
        self, tensor: Float[t.Tensor, "seq"], start_token: float
    ) -> Float[t.Tensor, "seq"]:
        """Shift the tensor to the right and insert the start token."""
        start_token = t.tensor([start_token], device=tensor.device)
        shifted_tensor = t.cat([start_token, tensor[:-1]], dim=0)
        return shifted_tensor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, t.Tensor]:
        item = self.dataset[idx]

        print(f"Untokenized smiles: {item['smiles']}")
        smiles = self._tokenize(item["smiles"])
        print(f"Tokenized smiles: {smiles}")

        mz, intensity = self._pad_ms(item["mzs"], item["intensities"])

        tgt_intensity = self._shift_right(intensity, self.intensity_start_token)
        tgt_mz = self._shift_right(mz, self.mz_start_token)

        return {
            "smiles_ids": smiles.input_ids.squeeze(),  # 1 seq
            "attention_mask": smiles.attention_mask.squeeze(),  # 1 seq
            "collision_energy": t.tensor([item["collision_energy"]]),  # 1
            "instrument_type": t.tensor([item["instrument_type"]]),  # 1
            "tgt_intensity": tgt_intensity,  # seq-1
            "tgt_mz": tgt_mz,  # seq-1
            "intensity": intensity,  # seq-1
            "mz": mz,  # seq-1
        }

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
