import torch.nn as nn
import torch.functional as F


import torch.nn as nn
import torch


# class MS2MolWrapperGenerator(nn.Module):
#     def __init__(self, generator_model, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.generator = generator_model  # Pass in the generator model (e.g., CustomBARTModel)

#     def forward(self, ms_input, intensity, attention_mask=None):
#         # Forward pass through the generator to convert MS/MS to SMILES
#         smiles_output = self.generator(
#             encoder_input_ids=ms_input, 
#             intensity=intensity, 
#             decoder_input_ids=ms_input,  
#             encoder_attention_mask=attention_mask
#         )
#         return smiles_output
    


class MS2MolWrapperDiscriminator(nn.Module):
    def __init__(self, base_model, selected_layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model  # Pass in the base transformer model (e.g., AutoModel)
        self.selected_layer = selected_layer  # String to select the named module
        
        # Classifier layers for real/fake decision
        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # Output a single real/fake classification
        )

    def forward(self, ms_data):
        # Dynamically select the named module output
        try:
            selected_module = getattr(self.base_model, self.selected_layer)
            hidden_state = selected_module(ms_data).last_hidden_state
        except AttributeError:
            raise ValueError(f"The module '{self.selected_layer}' does not exist in the base model.")

        # Global average pooling and pass through classifier
        validity = self.classifier(hidden_state.mean(dim=1))  # Global average pooling
        return validity


class Mol2MSDiscriminator(nn.Module):
    def __init__(self, base_model, selected_layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model  # Pass in the base transformer model (e.g., AutoModel)
        self.selected_layer = selected_layer  # String to select the named module

        # Classifier layers for real/fake decision
        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # Output a single real/fake classification
        )

    def forward(self, smiles_data):
        # Dynamically select the named module output
        try:
            selected_module = getattr(self.base_model, self.selected_layer)
            hidden_state = selected_module(smiles_data).last_hidden_state
        except AttributeError:
            raise ValueError(f"The module '{self.selected_layer}' does not exist in the base model.")

        # Global average pooling and pass through classifier
        validity = self.classifier(hidden_state.mean(dim=1))  # Global average pooling
        return validity



# class Mol2MSWrapperGenerator(nn.Module):
#     def __init__(self, generator_model, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.generator = generator_model  # Pass in the generator model (e.g., CustomBARTModel)

#     def forward(self, x, attention_mask=None):
#         # Forward pass through the generator to convert MS/MS to SMILES
#         self.generator
#         return smiles_output
    


