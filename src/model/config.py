"""Configuration for Hugging Face BartModel wrapper."""

from dataclasses import dataclass


@dataclass
class Mol2MSModelConfig:
    encoder_name: str = "gayane/BARTSmiles"
    collision_energy_dim: int = 87
    instrument_type_dim: int = 51
    max_length: int = 128
    max_ms_length: int = 512
    num_heads: int = 16
    num_layers: int = 12
