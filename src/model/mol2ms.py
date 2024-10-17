"""Custom Mol2MS model inspired by BART for MS/MS prediction from SMILES."""

import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
from transformers import AutoModel, AutoTokenizer

from .config import Mol2MSModelConfig


class Mol2MSModel(nn.Module):
    def __init__(self, config: Mol2MSModelConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_name)
        self.encoder = AutoModel.from_pretrained(self.config.encoder_name).encoder

        # Custom decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.encoder.config.hidden_size, nhead=self.config.num_heads
            ),
            num_layers=self.config.num_layers,
        )

        # Output layers
        self.continuous_output1 = nn.Linear(self.encoder.config.hidden_size, 1)
        self.continuous_output2 = nn.Linear(self.encoder.config.hidden_size, 1)
        self.binary_output = nn.Linear(self.encoder.config.hidden_size, 1)

        # Embeddings
        self.index_embedding = nn.Embedding(
            self.config.max_ms_length, self.encoder.config.hidden_size
        )
        self.collision_energy_embedding = nn.Embedding(
            self.config.collision_energy_dim, self.encoder.config.hidden_size
        )
        self.instrument_type_embedding = nn.Embedding(
            self.config.instrument_type_dim, self.encoder.config.hidden_size
        )

        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = "<s>"
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "</s>"

    def forward(
        self,
        tokenized_smiles: Int[t.Tensor, "batch seq"],
        attention_mask: Int[t.Tensor, "batch seq"],
        index: Int[t.Tensor, "batch 1"],
        collision_energy: Int[t.Tensor, "batch 1"],
        instrument_type: Int[t.Tensor, "batch 1"],
    ) -> tuple[
        Float[t.Tensor, "batch"],
        Float[t.Tensor, "batch"],
        Float[t.Tensor, "batch"],
    ]:
        smiles_encoding = self.encoder(
            input_ids=tokenized_smiles, attention_mask=attention_mask
        )

        index_embedding = self.index_embedding(index).transpose(0, 1)
        collision_energy_embedding = self.collision_energy_embedding(collision_energy).transpose(0, 1)
        instrument_type_embedding = self.instrument_type_embedding(instrument_type).transpose(0, 1)

        decoder_input = index_embedding + collision_energy_embedding + instrument_type_embedding

        decoder_output = self.decoder(
            tgt=decoder_input, memory=smiles_encoding.last_hidden_state.transpose(0, 1)
        ).transpose(0, 1)

        continuous_output1 = self.continuous_output1(decoder_output).squeeze(-1)
        continuous_output2 = self.continuous_output2(decoder_output).squeeze(-1)
        binary_output = t.sigmoid(self.binary_output(decoder_output)).squeeze(-1)

        return continuous_output1, continuous_output2, binary_output

    def tokenize(self, smiles_string: str):
        """Tokenizes a SMILES string using the BART tokenizer."""
        return self.tokenizer(
            smiles_string,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
        )

    def save(self, path: str):
        """Saves the model and tokenizer to the specified path."""
        t.save(self.state_dict(), f"{path}/model.pt")
        self.tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved to {path}")
