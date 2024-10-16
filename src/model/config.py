"""Configuration for Hugging Face BartModel wrapper."""

from dataclasses import dataclass


@dataclass
class Mol2MSModelConfig:
    encoder_name: str = "gayane/BARTSmiles"
    max_length: int = 128
    max_ms_length: int = 1000
    num_heads: int = 16
    num_layers: int = 12
