"""Custom Mol2MS model inspired by BART for MS/MS prediction from SMILES."""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from transformers import AutoModel, AutoTokenizer

from .config import BartModelConfig


class Mol2MSModel(nn.Module):
    def __init__(self, config: BartModelConfig = BartModelConfig()):
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

        # Embedding for integer input
        self.index_embedding = nn.Embedding(
            self.config.max_ms_length, self.encoder.config.hidden_size
        )

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        attention_mask: Int[torch.Tensor, "batch seq"],
        index: Int[torch.Tensor, "batch ms_seq"],
    ) -> tuple[
        Float[torch.Tensor, "batch"],
        Float[torch.Tensor, "batch"],
        Float[torch.Tensor, "batch"],
    ]:
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        decoder_input = self.index_embedding(index).transpose(0, 1)

        decoder_outputs = self.decoder(
            tgt=decoder_input, memory=encoder_outputs.last_hidden_state.transpose(0, 1)
        ).transpose(0, 1)

        continuous_output1 = self.continuous_output1(decoder_outputs).squeeze(-1)
        continuous_output2 = self.continuous_output2(decoder_outputs).squeeze(-1)
        binary_output = torch.sigmoid(self.binary_output(decoder_outputs)).squeeze(-1)

        return continuous_output1, continuous_output2, binary_output

    def tokenize(self, smiles_string: str):
        """Tokenizes a SMILES string using the BART tokenizer."""
        return self.tokenizer(
            smiles_string, return_tensors="pt", padding=True, truncation=True
        )

    def save(self, path: str):
        """Saves the model and tokenizer to the specified path."""
        torch.save(self.state_dict(), f"{path}/model.pt")
        self.tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved to {path}")
