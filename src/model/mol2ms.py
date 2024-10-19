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

        self.tokenizer = tokenizer

        # Encoder from BartSmiles
        self.encoder = AutoModel.from_pretrained(self.config.encoder_name).encoder
        self.encoder.resize_token_embeddings(len(self.tokenizer))

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
        self.pressure_embedding = nn.Embedding(
            self.config.pressure_dim, self.d_model
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
        pressure: Int[t.Tensor, "batch"],
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
        pressure_embedding = self.pressure_embedding(pressure).expand_as(
            smiles_embedding
        )  # batch seq d_model

        return (
            smiles_embedding
            + collision_energy_embedding
            + instrument_type_embedding
            + pressure_embedding
        )

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

    def _encoder_forward(
        self,
        smiles_ids: Int[t.Tensor, "batch seq"],
        attention_mask: Int[t.Tensor, "batch seq"],
        collision_energy: Int[t.Tensor, "batch"],
        instrument_type: Int[t.Tensor, "batch"],
        pressure: Int[t.Tensor, "batch"],
    ) -> Float[t.Tensor, "seq batch d_model"]:
        # Encoder embeddings
        encoder_input_embeddings = self._get_encoder_embeddings(
            smile_ids=smiles_ids,
            collision_energy=collision_energy,
            instrument_type=instrument_type,
            pressure=pressure,
        ).transpose(
            0, 1
        )  # seq batch d_model

        # Forward pass
        smiles_encoding = self.encoder(
            inputs_embeds=encoder_input_embeddings,
            attention_mask=attention_mask.transpose(0, 1),
        )  # seq batch d_model
        memory = smiles_encoding.last_hidden_state  # seq batch d_model

        return memory

    def _decoder_forward(
        self,
        tgt_intensities: Float[t.Tensor, "batch seq-1"],
        tgt_mzs: Float[t.Tensor, "batch ms_seq-1"],
        memory: Float[t.Tensor, "seq batch d_model"],
    ) -> Float[t.Tensor, "batch seq-1"]:
        batch, _ = tgt_intensities.shape
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

        decoder_output = self.decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask
        ).transpose(
            0, 1
        )  # batch seq d_model

        return decoder_output

    def forward(
        self,
        smiles_ids: Int[t.Tensor, "batch seq"],
        attention_mask: Int[t.Tensor, "batch seq"],
        collision_energy: Int[t.Tensor, "batch"],
        instrument_type: Int[t.Tensor, "batch"],
        pressure: Int[t.Tensor, "batch"],
        tgt_intensities: Float[t.Tensor, "batch seq-1"],
        tgt_mzs: Float[t.Tensor, "batch ms_seq-1"],
    ) -> tuple[Float[t.Tensor, "batch seq-1"], Float[t.Tensor, "batch seq-1"]]:
        memory = self._encoder_forward(
            smiles_ids=smiles_ids,
            attention_mask=attention_mask,
            collision_energy=collision_energy,
            instrument_type=instrument_type,
            pressure=pressure,
        )
        decoder_output = self._decoder_forward(
            tgt_intensities=tgt_intensities, tgt_mzs=tgt_mzs, memory=memory
        )

        mz_output = self.mz_output(decoder_output[:, 1:, :]).squeeze(-1)  # batch seq-1
        intensity_output = self.intensity_output(decoder_output[:, 1:, :]).squeeze(
            -1
        )  # batch seq-1

        return mz_output, intensity_output

    def inference(
        self, smiles: str, collision_energy: int, instrument_type: int, pressure: int
    ) -> tuple[Float[t.Tensor, "ms_seq"], Float[t.Tensor, "ms_seq"]]:
        """Inference for a single SMILES string."""
        device = next(self.parameters()).device

        smiles = t.tensor(smiles, device=device).unsqueeze(0)
        collision_energy = t.tensor(collision_energy, device=device).unsqueeze(0)
        instrument_type = t.tensor(instrument_type, device=device).unsqueeze(0)
        pressure = t.tensor(pressure, device=device).unsqueeze(0)

        smiles_ids = self.tokenizer(smiles, return_tensors="pt").input_ids

        attention_mask = t.ones_like(smiles_ids, device=device)

        memory = self._encoder_forward(
            smiles_ids=smiles_ids.unsqueeze(0),
            attention_mask=attention_mask,
            collision_energy=collision_energy,
            instrument_type=instrument_type,
            pressure=pressure,
        )

        mz_tensor, intensity_tensor = t.tensor([0.0], device=device), t.tensor(
            [0.0], device=device
        )

        while mz_tensor[-1] >= 0.0 and intensity_tensor[-1] >= 0.0:
            decoder_output = self._decoder_forward(
                tgt_intensities=intensity_tensor.unsqueeze(-1),
                tgt_mzs=mz_tensor.unsqueeze(-1),
                memory=memory,
            ).transpose(0, 1)

            mz_tensor = t.cat(
                [mz_tensor, self.mz_output(decoder_output[:, -1, :]).squeeze(0)], dim=0
            )
            intensity_tensor = t.cat(
                [
                    intensity_tensor,
                    self.intensity_output(decoder_output[:, -1, :]).squeeze(0),
                ],
                dim=0,
            )

        return mz_tensor, intensity_tensor

    def save(self, path: str, name: str):
        """Saves the model to the specified path."""
        t.save(self.state_dict(), f"{path}/{name}.pt")
        print(f"Model saved to {path}/{name}.pt")
