from torch.nn import Module

import torch
import torch.nn as nn
from transformers import BartModel, BartTokenizer, AdamW

class CustomBARTModel(nn.Module):
    def __init__(self, bart_model):
        super().__init__()
        self.bart = bart_model
        
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