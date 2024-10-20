"""Configuration for Hugging Face BartModel wrapper."""

from dataclasses import dataclass


@dataclass
class Mol2MSModelConfig:
    encoder_name: str = "gayane/BARTSmiles"
    collision_energy_dim: int = 87
    instrument_type_dim: int = 51
    dropout: float = 0.1
    max_encoder_length: int = 128
    max_decoder_length: int = 512
    num_heads: int = 16
    num_layers: int = 12
    hidden_size:int=512
    
@dataclass
class MS2MolModelConfig:
    encoder_name: str = "gayane/BARTSmiles"  # Pretrained encoder model
    collision_energy_dim: int = 87  # Dimension for collision energy
    instrument_type_dim: int = 51  # Dimension for instrument type
    max_length: int = 128  # Maximum length for SMILES or decoder input
    max_ms_length: int = 512  # Maximum length for ms/z sequence
    num_heads: int = 16  # Number of attention heads
    num_layers: int = 12  # Number of layers in the encoder/decoder
    d_model: int = 512  # Hidden dimension size (embedding size)
    smiles_vocab_size: int = 1026  # Vocabulary size for SMILES
    mz_vocab_size: int = 512  # Output size for m/z prediction
    machine_type_vocab_size: int = 51  # Number of machine types
    hidden_size:int= 512

# #TODO: rename this
# @dataclass
# class DeMy:
#     mol_ms_config: Mol2MSModelConfig=Mol2MSModelConfig()
#     ms_mol_config: MS2MolModelConfig =MS2MolModelConfig()
#     other:dict = (arbitraryDict:={})
    