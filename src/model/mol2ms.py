"""Custom Mol2MS model inspired by BART for MS/MS prediction from SMILES."""

import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
from transformers import AutoModel, AutoTokenizer

from src.embeddings.mz_pos_embedding import MZPositionalEncoding

from .config import Mol2MSModelConfig


class Mol2MSModel(nn.Module):
    def __init__(self, config: Mol2MSModelConfig, tokenizer: AutoTokenizer):
        super().__init__()
        self.config = config

        # Encoder from BartSmiles
        self.encoder = AutoModel.from_pretrained(self.config.encoder_name).encoder
        self.encoder.resize_token_embeddings(len(tokenizer))

        self.d_model = self.encoder.config.hidden_size

        # Custom decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.config.dropout,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        # Encoder embeddings
        self.encoder_embeddings = self.encoder.get_input_embeddings()
        self.collision_energy_embedding = nn.Embedding(
            self.config.collision_energy_dim, self.d_model
        )
        self.instrument_type_embedding = nn.Embedding(
            self.config.instrument_type_dim, self.d_model
        )

        # Decoder embeddings
        self.decoder_cls_token = nn.Parameter(t.randn(1, 1, self.d_model))
        self.intensity_embedding = nn.Linear(1, self.d_model)
        self.mz_positional_embedding = MZPositionalEncoding(
            d_model=self.d_model, freq_scale=1.0, normalize=False
        )

        # Output layers
        self.mz_output = nn.Linear(self.d_model, 1)
        self.intensity_output = nn.Linear(self.d_model, 1)

    def _get_encoder_embeddings(
        self,
        smile_ids: Int[t.Tensor, "batch seq"],
        collision_energy: Int[t.Tensor, "batch"],
        instrument_type: Int[t.Tensor, "batch"],
    ) -> Float[t.Tensor, "batch seq d_model"]:
        """Get the encoder embeddings for the SMILES."""
        smiles_embedding = self.encoder_embeddings(smile_ids)  # batch seq d_model
        collision_energy_embedding = self.collision_energy_embedding(
            collision_energy
        ).expand_as(
            smiles_embedding
        )  # batch seq d_model
        instrument_type_embedding = self.instrument_type_embedding(
            instrument_type
        ).expand_as(
            smiles_embedding
        )  # batch seq d_model

        return smiles_embedding + collision_energy_embedding + instrument_type_embedding

    def _get_decoder_embeddings(
        self,
        target_intensities: Float[t.Tensor, "batch seq-1"],
        target_mzs: Float[t.Tensor, "batch seq-1"],
    ) -> Float[t.Tensor, "batch seq-1 d_model"]:
        """Get the decoder embeddings for the target intensities and mzs."""
        intensity_embedding = self.intensity_embedding(
            target_intensities
        )  # batch seq-1 d_model
        mz_embedding = self.mz_positional_embedding(target_mzs)  # batch seq-1 d_model

        return intensity_embedding + mz_embedding

    def forward(
        self,
        smiles_ids: Int[t.Tensor, "batch seq"],
        attention_mask: Int[t.Tensor, "batch seq"],
        collision_energy: Int[t.Tensor, "batch"],
        instrument_type: Int[t.Tensor, "batch"],
        tgt_intensities: Float[t.Tensor, "batch seq-1"],
        tgt_mzs: Float[t.Tensor, "batch ms_seq-1"],
    ) -> tuple[Float[t.Tensor, "batch seq-1"], Float[t.Tensor, "batch seq-1"]]:
        batch, seq = smiles_ids.shape

        # Encoder embeddings
        encoder_input_embeddings = self._get_encoder_embeddings(
            smile_ids=smiles_ids,
            collision_energy=collision_energy,
            instrument_type=instrument_type,
        ).transpose(
            0, 1
        )  # seq batch d_model

        # Decoder embeddings
        decoder_embeddings = self._get_decoder_embeddings(
            target_intensities=tgt_intensities.unsqueeze(-1), target_mzs=tgt_mzs
        )  # batch seq d_model

        decoder_cls_token = self.decoder_cls_token.expand(
            batch, 1, self.d_model
        )  # batch 1 d_model
        tgt = t.cat([decoder_cls_token, decoder_embeddings], dim=1).transpose(
            0, 1
        )  # seq batch d_model
        tgt_mask = t.triu(
            t.full(
                (self.config.max_decoder_length, self.config.max_decoder_length),
                float("-inf"),
            ),
            diagonal=1,
        ).to(
            tgt.device
        )  # seq seq

        # Forward pass
        smiles_encoding = self.encoder(
            inputs_embeds=encoder_input_embeddings,
            attention_mask=attention_mask.transpose(0, 1),
        )  # seq batch d_model
        memory = smiles_encoding.last_hidden_state  # seq batch d_model

        decoder_output = self.decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask
        ).transpose(
            0, 1
        )  # batch seq d_model

        mz_output = self.mz_output(decoder_output[:, 1:, :]).squeeze(-1)  # batch seq-1
        intensity_output = self.intensity_output(decoder_output[:, 1:, :]).squeeze(
            -1
        )  # batch seq-1

        return mz_output, intensity_output

    def save(self, path: str):
        """Saves the model to the specified path."""
        t.save(self.state_dict(), f"{path}/model.pt")
        print(f"Model saved to {path}")
