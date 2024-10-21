from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# TODO: include the EV stuff.
class Mol2MSDataset(Dataset):
    def __init__(self, hf_dataset: Dataset, model_name: str, max_length: int):
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


class MS2MolDataset(Dataset):
    def __init__(
        self,
        exact_mass,
        encoder_spectrum_preTK,
        decoder_smiles_preTK,
        encoder_tokenizer,
        decoder_tokenizer,
    ):
        self.exact_mass = exact_mass
        self.encoder_spectrum_preTK = encoder_spectrum_preTK
        self.decoder_smiles_preTK = decoder_smiles_preTK
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

    def __len__(self):
        return len(self.encoder_spectrum_preTK)

    def str_to_float(str_spectrum):
        # print(f"len str_spec: {len(str_spectrum)}")
        float_spectrum = []

        left, right = 1, 0
        # print(len(str_spectrum))
        while right < len(str_spectrum) - 1:

            # print(str_spectrum[right])
            if str_spectrum[right] == "]":
                while str_spectrum[left] != "[":
                    left += 1
                str_tuple = str_spectrum[left + 1 : right]
                x, y = str_tuple.split(",")
                float_spectrum.append([np.round(float(x), 1), float(y)])
                # print(x, y)
                left = right

                right += 1
            right += 1

        # print(len(float_spectrum))
        _, float_intensity_tuple = zip(*float_spectrum)
        return float_intensity_tuple

    def __getitem__(self, idx):

        float_exact_mass = np.round(self.exact_mass[idx], 1)
        float_spectrum = str_to_float(self.encoder_spectrum_preTK[idx])
        # if idx == 0:
        # print(len(self.encoder_spectrum_preTK[idx]))
        # print(len(float_spectrum))
        _, float_intensity_tuple = zip(*float_spectrum)
        float_intensity_list = list(float_intensity_tuple)
        # print(float_spectrum)
        # print(float_intensity)
        float_intensity_list.insert(
            0, 2.0
        )  # for the exact mass, doens't have theoerical intensity so must use 2.0
        float_intensity_list_padded = float_intensity_list
        if len(float_intensity_list) == 257:
            float_intensity_list.pop()
        for i in range(256 - len(float_intensity_list)):
            if len(float_intensity_list) == 256:
                break
            float_intensity_list_padded.append(1)
        encoder_mass_tokenized = self.encoder_tokenizer(
            float_exact_mass, float_spectrum
        )  # haven't turned data into
        encoder_mass_tokenized_tensor = torch.tensor(encoder_mass_tokenized)
        # float yet
        encoder_attention_mask = (encoder_mass_tokenized_tensor != 1).long()
        # encoder_attention_mask_with_random = random_mask_exact_mass(encoder_attention_mask)
        encoder_attention_mask_tensor = torch.tensor(encoder_attention_mask)

        decoder_smiles_tokenized, decoder_label = self.decoder_tokenizer(
            self.decoder_smiles_preTK[idx]
        )
        decoder_label_tensor = torch.tensor(decoder_label)
        decoder_smiles_tokenized_tensor = torch.tensor(decoder_smiles_tokenized)

        decoder_attention_mask = (decoder_smiles_tokenized_tensor != 1).long()

        decoder_attention_mask_tensor = torch.tensor(decoder_attention_mask)

        # print(f"len enc tens:{len(encoder_mass_tokenized)}")
        # print(f"len att:{len(encoder_attention_mask)}")
        # print(f"len dec tens:{len(decoder_smiles_tokenized)}")
        # print(f"len att:{len(decoder_attention_mask)}")

        # print(f"len intens:{len(float_intensity_list)}")
        return {
            "encoder_input_id": encoder_mass_tokenized_tensor,
            "encoder_attention_mask": encoder_attention_mask_tensor,
            "decoder_input_id": decoder_smiles_tokenized_tensor,
            "decoder_attention_mask": decoder_attention_mask_tensor,
            "decoder_label": decoder_label_tensor,
            "intensity": torch.tensor(float_intensity_list),
        }


# class MS2MolDatasetRewrite(Dataset):
#     def __init__(self):
#         super().__init__()

#     def
