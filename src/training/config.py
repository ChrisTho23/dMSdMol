"""Configuration for distributed training on SageMaker."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Mol2MSTrainingConfig:
    dataset_name: str = field(
        default="ChrisTho/mol2ms_enveda",  # ChrisTho/dMSdMol_dummy_data
        metadata={"help": "Name of the dataset to use for training"},
    )
    batch_size: int = field(default=32, metadata={"help": "Batch size for training"})
    learning_rate: float = field(
        default=5e-5, metadata={"help": "Learning rate for training"}
    )
    num_epochs: int = field(default=3, metadata={"help": "Number of epochs to train"})
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for learning rate scheduler"},
    )
    max_length: int = field(
        default=128, metadata={"help": "Maximum sequence length for tokenization"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision training"},
    )


@dataclass
class SageMakerTrainingConfig:
    aws_region: str = field(
        default="us-east-1", metadata={"help": "AWS region for SageMaker training"}
    )
    entry_point: str = field(
        default="src/training/train_script.py",
        metadata={"help": "Entry point script for SageMaker training job"},
    )
    source_dir: str = field(
        default="./", metadata={"help": "Source directory containing the training code"}
    )
    instance_type: str = field(
        default="ml.p3.16xlarge", metadata={"help": "EC2 instance type for training"}
    )
    instance_count: int = field(
        default=1, metadata={"help": "Number of EC2 instances to use for training"}
    )
    output_dir: str = field(
        default="/opt/ml/output",
        metadata={"help": "Output directory for SageMaker training job"},
    )
    transformers_version: str = field(
        default="4.28.1", metadata={"help": "Version of Transformers library to use"}
    )
    pytorch_version: str = field(
        default="2.0.0", metadata={"help": "Version of PyTorch to use"}
    )
    py_version: str = field(default="py310", metadata={"help": "Python version to use"})
    git_config: Dict[str, str] = field(
        default_factory=lambda: {
            "repo": "https://github.com/ChrisTho23/dMSdMol",
            "branch": "fix-dp",
        },
        metadata={"help": "Git configuration for source code"},
    )
    distribution: Dict[str, Dict[str, bool]] = field(
        default_factory=lambda: {"smdistributed": {"dataparallel": {"enabled": True}}},
        metadata={"help": "Distribution configuration for SageMaker training"},
    )
    hyperparameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "wandb_project": os.getenv("WANDB_PROJECT"),
            "wandb_api_key": os.getenv("WANDB_API_KEY"),
        },
        metadata={"help": "Hyperparameters for training"},
    )
    dependencies: List[str] = field(
        default_factory=lambda: ["requirements.txt"],
        metadata={"help": "Environment variables for SageMaker training"},
    )
    image_uri: str = field(
        default="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",
        metadata={"help": "Image URI for SageMaker training"},
    )
