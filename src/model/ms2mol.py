from torch.nn import Module

import torch
import torch.nn as nn
from transformers import BartModel, BartTokenizer, AdamW

class CustomBARTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bart = BartModel()
        
        # Separate embedding layers for encoder and decoder
        self.encoder_embedding = nn.Embedding(10000 + 3, 128)
        self.decoder_embedding = nn.Embedding(269 + 3, 128)
        # Replace the default shared embeddings in BART
        self.bart.encoder.embed_tokens = self.encoder_embedding
        self.bart.decoder.embed_tokens = self.decoder_embedding
        
        # Fully connected layers for encoder
        # self.fc1 = nn.Linear(769, 768)
        # self.fc2 = nn.Linear(768, 768)
        self.fc1 = nn.Linear(129, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_logits = torch.nn.Linear(self.bart.config.d_model,269 + 3 )



    def forward(self, encoder_input_ids, intensity, decoder_input_ids, encoder_attention_mask=None, decoder_attention_mask=None):
        # Encoder: Get embeddings and concatenate extra tensor
        encoder_embedded = self.encoder_embedding(encoder_input_ids)
        intensity = intensity.unsqueeze(-1)
        combined_encoder_embedded = torch.cat((encoder_embedded, intensity), dim=-1)
        combined_encoder_embedded = self.fc1(combined_encoder_embedded)
        combined_encoder_embedded = torch.relu(combined_encoder_embedded)
        combined_encoder_embedded = self.fc2(combined_encoder_embedded)

        
        
        # Forward pass through BART with modified embeddings

        encoder_outputs = self.bart.encoder(
            inputs_embeds=combined_encoder_embedded,
            attention_mask=encoder_attention_mask
        )
        
        # Decoder: Get embeddings
        decoder_embedded = self.decoder_embedding(decoder_input_ids)
        
        # Pass through the decoder
        decoder_outputs = self.bart.decoder(
            inputs_embeds=decoder_embedded,
            encoder_hidden_states=encoder_outputs[0],  # Pass encoder outputs to decoder
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask
        )
        logits = self.fc_logits(decoder_outputs.last_hidden_state)
        return logits
    

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel



class MS2MolModel(nn.Module):
    def __init__(self,  config):
        super(MS2MolModel, self).__init__()
        self.config = config

        # Load the BART model
        self.bart = BartModel(config)
        
        # Embedding for individual m/z values (each m/z is its own token)
        self.encoder_embedding = nn.Linear(1, self.config.d_model)  # Project each m/z value into d_model space
        
        # Embedding for intensities (also tokenized individually)
        self.intensity_embedding = nn.Linear(1, self.config.d_model)  # Project each intensity into d_model space

        # Embedding for the decoder (for SMILES sequence)
        self.decoder_embedding = nn.Embedding(self.config.smiles_vocab_size, self.config.d_model)

        # Replace the default shared embeddings in BART
        self.bart.encoder.embed_tokens = None  # Since we use continuous values for tokens
        self.bart.decoder.embed_tokens = self.decoder_embedding

        # Output layers for different tasks
        self.fc_logits = nn.Linear(self.config.d_model, self.config.smiles_vocab_size)  # SMILES output
        self.collision_energy_output = nn.Linear(self.config.d_model, 1)  # Collision energy prediction
        self.machine_type_output = nn.Linear(self.config.d_model, self.config.machine_type_vocab_size)  # Machine type prediction

    def _get_encoder_embeddings(self, encoder_input_ids, intensity):
        """Get the encoder embeddings by combining m/z values and intensities for each token."""
        # Expand the dimension of each ms/z and intensity so they can be embedded
        encoder_embedded = self.encoder_embedding(encoder_input_ids.unsqueeze(-1))  # Shape: (batch_size, 512, d_model)
        intensity_embedded = self.intensity_embedding(intensity.unsqueeze(-1))  # Shape: (batch_size, 512, d_model)

        # Combine the embeddings (element-wise addition of ms/z and intensity embeddings)
        combined_embeddings = encoder_embedded + intensity_embedded  # Shape: (batch_size, 512, d_model)

        return combined_embeddings

    def forward(self, encoder_input_ids, intensity, decoder_input_ids, encoder_attention_mask=None, decoder_attention_mask=None):
        # Get encoder embeddings with intensity added
        encoder_embedded = self._get_encoder_embeddings(encoder_input_ids, intensity)  # Shape: (batch_size, 512, d_model)
        
        # Pass through the BART encoder
        encoder_outputs = self.bart.encoder(
            inputs_embeds=encoder_embedded,  # Shape: (batch_size, 512, d_model)
            attention_mask=encoder_attention_mask
        )
        
        # Get decoder embeddings for SMILES sequence
        decoder_embedded = self.decoder_embedding(decoder_input_ids)  # Shape: (batch_size, seq_length, d_model)
        
        # Pass through the BART decoder
        decoder_outputs = self.bart.decoder(
            inputs_embeds=decoder_embedded,
            encoder_hidden_states=encoder_outputs[0],  # Shape: (batch_size, 512, d_model)
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False
        )
        
        # Output for SMILES (m/z values)
        smiles_output = self.fc_logits(decoder_outputs.last_hidden_state)  # Shape: (batch_size, seq_length, smiles_vocab_size)

        # Additional heads for collision energy and machine type prediction
        collision_energy_output = self.collision_energy_output(decoder_outputs.last_hidden_state.mean(dim=1))  # Mean pooling, Shape: (batch_size, 1)
        machine_type_output = self.machine_type_output(decoder_outputs.last_hidden_state.mean(dim=1))  # Mean pooling, Shape: (batch_size, machine_type_vocab_size)
        
        return smiles_output, collision_energy_output, machine_type_output
    def generate(
        self,
        encoder_input_ids,
        intensity,
        max_length=50,
        encoder_attention_mask=None,
        eos_token_id=None,
        store_gradients=False
    ):
        """Generate SMILES autoregressively using greedy decoding, with optional gradient storage."""
        
        batch_size = encoder_input_ids.size(0)
        device = encoder_input_ids.device
        
        # Use gradient computation if `store_gradients` is True, otherwise disable gradients
        context_manager = t.enable_grad() if store_gradients else t.no_grad()
        
        # Perform generation within the context manager
        with context_manager:
            # Get encoder embeddings with intensity added
            encoder_embedded = self._get_encoder_embeddings(encoder_input_ids, intensity)
            
            # Pass through the encoder
            encoder_outputs = self.bart.encoder(
                inputs_embeds=encoder_embedded,
                attention_mask=encoder_attention_mask
            )
            
            # Prepare decoder inputs, starting with the <CLS> token (assumed to be token id 0)
            decoder_input_ids = t.zeros((batch_size, 1), dtype=t.long, device=device)
            
            generated_tokens = decoder_input_ids  # Collect generated tokens
            
            # Autoregressive generation loop (greedy decoding)
            for _ in range(max_length):
                # Get decoder embeddings for the current step
                decoder_embedded = self.decoder_embedding(decoder_input_ids)
                
                # Forward pass through the decoder
                decoder_outputs = self.bart.decoder(
                    inputs_embeds=decoder_embedded,
                    encoder_hidden_states=encoder_outputs[0],
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=False
                )
                
                # Compute the logits for the last token
                logits = self.fc_logits(decoder_outputs.last_hidden_state[:, -1, :])  # Take last token's logits
                
                # Greedy decoding: take the argmax of the logits
                next_token = t.argmax(logits, dim=-1).unsqueeze(-1)  # Get the next token (batch_size, 1)
                
                # Append the predicted token to the decoder inputs
                decoder_input_ids = t.cat([decoder_input_ids, next_token], dim=1)
                generated_tokens = t.cat([generated_tokens, next_token], dim=1)
                
                # Break if end token is generated
                if eos_token_id is not None and t.any(next_token == eos_token_id):
                    break
        
        return generated_tokens