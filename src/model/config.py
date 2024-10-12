"""Configuration for Hugging Face BartModel wrapper."""

from dataclasses import dataclass


@dataclass
class BartModelConfig:
    model_name: str = "facebook/bart-large"
    max_length: int = 128
    output_dir: str = "./bart-chem-model"
