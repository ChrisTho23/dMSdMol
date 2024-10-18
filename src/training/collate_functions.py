
from torch.nn.utils.rnn import pad_sequence
import torch


def collate_fn_msz(batch):
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )
    mz = torch.stack([item["mz"] for item in batch])
    intensity = torch.stack([item["intensity"] for item in batch])
    index = torch.stack([item["index"] for item in batch])
    create_next_token = torch.stack([item["create_next_token"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mz": mz,
        "intensity": intensity,
        "index": index,
        "create_next_token": create_next_token,
    }


def collate_fn_cycleGan(batch):
    """this function should essentially build the following, 

    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """   


   return {"input_ids": }