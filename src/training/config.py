"""Configuration for distributed training on SageMaker."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Mol2MSTrainingConfig:
    dataset_name: str = "ChrisTho/dMSdMol_dummy_data"
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 0
    save_every: int = 0
    model_dir: str = "./logs/"
    max_length: int = 128
    seed: int = 42
    fp16: bool = False

@dataclass
class SageMakerTrainingConfig:
    output_dir: str = "/opt/ml/output"
    entry_point: str = "train_script.py"
    source_dir: str = "./src/training"
    instance_type: str = "ml.p3.2xlarge"
    instance_count: int = 1
    transformers_version: str = '4.45'
    pytorch_version: str = '2.4'
    py_version: str = 'py310'
    git_config: Dict[str, str] = {"repo": "https://github.com/ChrisTho23/dMSdMol", "branch": "sagemaker-training"}
    distribution: Dict[str, Dict[str, bool]] = {"smdistributed": {"dataparallel": {"enabled": True}}}

