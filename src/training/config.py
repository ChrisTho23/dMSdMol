"""Configuration for distributed training on SageMaker."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Mol2MSTrainingConfig:
    dataset_name: str = field(
        default="ChrisTho/dMSdMol_dummy_data",
        metadata={"help": "Name of the dataset to use for training"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for training"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for training"}
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "Number of epochs to train"}
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for learning rate scheduler"}
    )
    save_every: int = field(
        default=0,
        metadata={"help": "Save model every X steps (0 to disable)"}
    )
    model_dir: str = field(
        default="./logs/",
        metadata={"help": "Directory to save model checkpoints"}
    )
    max_length: int = field(
        default=128,
        metadata={"help": "Maximum sequence length for tokenization"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision training"}
    )

@dataclass
class SageMakerTrainingConfig:
    output_dir: str = field(
        default="/opt/ml/output",
        metadata={"help": "Output directory for SageMaker training job"}
    )
    entry_point: str = field(
        default="train_script.py",
        metadata={"help": "Entry point script for SageMaker training job"}
    )
    source_dir: str = field(
        default="./src/training",
        metadata={"help": "Source directory containing the training code"}
    )
    instance_type: str = field(
        default="ml.p3.2xlarge",
        metadata={"help": "EC2 instance type for training"}
    )
    instance_count: int = field(
        default=1,
        metadata={"help": "Number of EC2 instances to use for training"}
    )
    transformers_version: str = field(
        default='4.4.2',
        metadata={"help": "Version of Transformers library to use"}
    )
    pytorch_version: str = field(
        default='1.6.0',
        metadata={"help": "Version of PyTorch to use"}
    )
    py_version: str = field(
        default='py36',
        metadata={"help": "Python version to use"}
    )
    git_config: Dict[str, str] = field(
        default_factory=lambda: {"repo": "https://github.com/ChrisTho23/dMSdMol", "branch": "sagemaker-training"},
        metadata={"help": "Git configuration for source code"}
    )
    distribution: Dict[str, Dict[str, bool]] = field(
        default_factory=lambda: {"smdistributed": {"dataparallel": {"enabled": True}}},
        metadata={"help": "Distribution configuration for SageMaker training"}
    )
    aws_region: str = field(
        default="us-east-1",
        metadata={"help": "AWS region for SageMaker training"}
    )
    hyperparameters: Dict[str, Any] = field(
        default_factory=lambda: {},
        metadata={"help": "Hyperparameters for training"}
    )
