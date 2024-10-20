
from torch.nn.utils.rnn import pad_sequence
import torch


def collate_fn(batch):
    tokenized_smiles = pad_sequence(
        [item["tokenized_smiles"] for item in batch], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )

    mz = torch.tensor([item["mz"] for item in batch], dtype=torch.float).unsqueeze(1)
    intensity = torch.tensor(
        [item["intensity"] for item in batch], dtype=torch.float
    ).unsqueeze(1)
    index = torch.tensor([item["index"] for item in batch], dtype=torch.long).unsqueeze(
        1
    )
    collision_energy = torch.tensor(
        [item["collision_energy"] for item in batch], dtype=torch.long
    ).unsqueeze(1)
    instrument_type = torch.tensor(
        [item["instrument_type"] for item in batch], dtype=torch.long
    ).unsqueeze(1)
    stop_token = torch.tensor(
        [item["stop_token"] for item in batch], dtype=torch.float
    ).unsqueeze(1)

    return {
        "tokenized_smiles": tokenized_smiles,
        "attention_mask": attention_mask,
        "index": index,
        "collision_energy": collision_energy,
        "instrument_type": instrument_type,
        "mz": mz,
        "intensity": intensity,
        "stop_token": stop_token,
    }


def collate_fn_cycleGan(batch):
    tokenized_smiles = pad_sequence(
        [item["tokenized_smiles"] for item in batch], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )

    mz = torch.tensor([item["mz"] for item in batch], dtype=torch.float).unsqueeze(1)
    intensity = torch.tensor(
        [item["intensity"] for item in batch], dtype=torch.float
    ).unsqueeze(1)
    index = torch.tensor([item["index"] for item in batch], dtype=torch.long).unsqueeze(
        1
    )
    collision_energy = torch.tensor(
        [item["collision_energy"] for item in batch], dtype=torch.long
    ).unsqueeze(1)
    instrument_type = torch.tensor(
        [item["instrument_type"] for item in batch], dtype=torch.long
    ).unsqueeze(1)
    stop_token = torch.tensor(
        [item["stop_token"] for item in batch], dtype=torch.float
    ).unsqueeze(1)

    return {
        "tokenized_smiles": tokenized_smiles,
        "attention_mask": attention_mask,
        "index": index,
        "collision_energy": collision_energy,
        "instrument_type": instrument_type,
        "mz": mz,
        "intensity": intensity,
        "stop_token": stop_token,
    }