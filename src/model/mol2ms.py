"""Custom Mol2MS model inspired by BART for MS/MS prediction from SMILES."""

import torch
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
            collision_energy.long()
        ).expand_as(
            smiles_embedding
        )  # batch seq d_model
        instrument_type_embedding = self.instrument_type_embedding(
            instrument_type.long()
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

    def _encoder_forward(
        self,
        smiles_ids: Int[t.Tensor, "batch seq"],
        attention_mask: Int[t.Tensor, "batch seq"],
        collision_energy: Int[t.Tensor, "batch"],
        instrument_type: Int[t.Tensor, "batch"],
    ) -> Float[t.Tensor, "seq batch d_model"]:
        # Encoder embeddings
        encoder_input_embeddings = self._get_encoder_embeddings(
            smile_ids=smiles_ids,
            collision_energy=collision_energy,
            instrument_type=instrument_type,
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
        tgt_intensities: Float[t.Tensor, "batch seq-1"],
        tgt_mzs: Float[t.Tensor, "batch ms_seq-1"],
    ) -> tuple[Float[t.Tensor, "batch seq-1"], Float[t.Tensor, "batch seq-1"]]:
        memory = self._encoder_forward(
            smiles_ids=smiles_ids,
            attention_mask=attention_mask,
            collision_energy=collision_energy,
            instrument_type=instrument_type,
        )
        decoder_output = self._decoder_forward(
            tgt_intensities=tgt_intensities, tgt_mzs=tgt_mzs, memory=memory
        )

        mz_output = self.mz_output(decoder_output[:, 1:, :]).squeeze(-1)  # batch seq-1
        intensity_output = self.intensity_output(decoder_output[:, 1:, :]).squeeze(
            -1
        )  # batch seq-1

        return mz_output, intensity_output

    def generate(
        self,
        encoder_input_ids,
        collision_energy,
        instrument_type,
        max_length=50,
        encoder_attention_mask=None,
        eos_token_id=None,
        store_gradients=False,
    ):
        """Generate m/z and intensity autoregressively using greedy decoding."""

        # Check if encoder_input_ids (SMILES) is a string, and if so, convert it to token IDs
        if isinstance(encoder_input_ids, str):
            encoder_input_ids = self.tokenizer.encode(
                encoder_input_ids, return_tensors="pt"
            )

        # Ensure inputs are on the correct device
        device = next(self.parameters()).device
        encoder_input_ids = encoder_input_ids.to(device)
        collision_energy = collision_energy.to(device)
        instrument_type = instrument_type.to(device)

        # Use gradient computation if `store_gradients` is True, otherwise disable gradients
        context_manager = torch.enable_grad() if store_gradients else torch.no_grad()

        with context_manager:
            # Get encoder embeddings with collision energy and instrument type added
            encoder_embedded = self._get_encoder_embeddings(
                smile_ids=encoder_input_ids,
                collision_energy=collision_energy,
                instrument_type=instrument_type,
            ).transpose(
                0, 1
            )  # Transpose for transformer format (seq, batch, d_model)

            # Pass through the encoder
            encoder_outputs = self.encoder(
                inputs_embeds=encoder_embedded, attention_mask=encoder_attention_mask
            )  # seq, batch, d_model

            memory = encoder_outputs.last_hidden_state  # seq, batch, d_model

            # Expand the already embedded CLS token to match the batch size
            decoder_cls_token = self.decoder_cls_token.expand(
                encoder_input_ids.size(0), -1, -1
            ).to(
                device
            )  # Shape: batch, 1, d_model

            # Initialize empty lists to hold generated m/z and intensity values
            generated_mz = []
            generated_intensities = []

            # Autoregressive generation loop (greedy decoding)
            for step in range(max_length):
                if step == 0:
                    # First step: Use the embedded CLS token directly for both m/z and intensity
                    prev_mz = decoder_cls_token  # Shape: batch, 1, d_model
                    prev_intensity = decoder_cls_token  # Shape: batch, 1, d_model
                    self.config.max_decoder_length

                    pad_mz = torch.zeros(
                        2,
                        self.config.max_decoder_length - 1,
                        self.d_model,
                        device=device,
                    )

                    padded_mz = torch.cat([prev_mz, pad_mz], dim=1)

                    decoder_output = self.decoder(
                        tgt=prev_mz.transpose(0, 1), memory=memory
                    ).transpose(
                        0, 1
                    )  # Shape: batch, seq, d_model
                    # Compute m/z and intensity outputs for this step
                    next_mz = self.mz_output(decoder_output[:, -1, :]).unsqueeze(
                        1
                    )  # Shape: batch, 1
                    next_intensity = self.intensity_output(
                        decoder_output[:, -1, :]
                    ).unsqueeze(
                        1
                    )  # Shape: batch, 1
                    generated_mz.append(next_mz)
                    generated_intensities.append(next_intensity)
                else:
                    # After the first step, use the previously generated outputs (which are embedded)
                    prev_mz = generated_mz[-1]  # Shape: batch, 1, d_model
                    prev_intensity = generated_intensities[
                        -1
                    ]  # Shape: batch, 1, d_model

                    # Embed the previously generated m/z and intensity using _get_decoder_embeddings
                    print(prev_mz.shape)
                    print(prev_intensity.shape)
                    decoder_embedded = self._get_decoder_embeddings(
                        target_intensities=prev_intensity,  # Already in the embedded space
                        target_mzs=prev_mz.squeeze(-1),  # Already in the embedded space
                    )  # Shape: batch, 1, d_model

                    # Concatenate decoder_cls_token only after the first step
                    # Shape: batch, seq, d_model

                    # Transpose to match the transformer input format (seq, batch, d_model)
                    tgt = decoder_embedded.transpose(0, 1)  # Shape: seq, batch, d_model

                    # Mask for autoregressive generation
                    tgt_mask = torch.triu(
                        torch.full((tgt.size(0), tgt.size(0)), float("-inf")),
                        diagonal=1,
                    ).to(device)

                    # Forward pass through the decoder
                    decoder_output = self.decoder(
                        tgt=tgt, memory=memory, tgt_mask=tgt_mask
                    ).transpose(
                        0, 1
                    )  # Shape: batch, seq, d_model

                    # Compute m/z and intensity outputs for this step
                    next_mz = self.mz_output(decoder_output[:, -1, :]).unsqueeze(
                        1
                    )  # Shape: batch, 1
                    next_intensity = self.intensity_output(
                        decoder_output[:, -1, :]
                    ).unsqueeze(
                        1
                    )  # Shape: batch, 1

                    # Append the predicted m/z and intensity for this step
                    generated_mz.append(next_mz)
                    generated_intensities.append(next_intensity)

                    # Optionally break if eos_token_id is generated (for SMILES generation, not m/z and intensity)
                    if eos_token_id is not None and torch.any(next_mz == eos_token_id):
                        break

            # Concatenate the generated m/z and intensity values along the sequence dimension
            generated_mz = torch.cat(generated_mz, dim=1)  # Shape: batch, seq, d_model
            generated_intensities = torch.cat(
                generated_intensities, dim=1
            )  # Shape: batch, seq, d_model

        return generated_mz, generated_intensities

    def save(self, path: str, name: str):
        """Saves the model to the specified path."""
        t.save(self.state_dict(), f"{path}/{name}.pt")
        print(f"Model saved to {path}/{name}.pt")
