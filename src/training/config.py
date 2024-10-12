"""Configuration for distributed training on SageMaker."""

from dataclasses import dataclass


@dataclass
class SageMakerTrainingConfig:
    dataset_name: str = "ChrisTho/dMSdMol_dummy_data"
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 0
    save_every: int = 1
    model_dir: str = "./logs/"
    max_length: int = 128
    seed: int = 42
    fp16: bool = False
