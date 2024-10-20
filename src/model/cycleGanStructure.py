import torch.nn as nn
import torch.functional as F


import torch.nn as nn
import torch


class MS2MolWrapperGenerator(nn.Module):
    def __init__(self, generator_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator_model  
        self.encoder = generator_model.encoder# Pass in the generator model (e.g., CustomBARTModel)
        self.decoder = generator_model.decoder
    def forward(self, ms_input, intensity, attention_mask=None):
        # Forward pass through the generator to convert MS/MS to SMILES
        smiles_output = self.generator(
            encoder_input_ids=ms_input, 
            intensity=intensity, 
            decoder_input_ids=ms_input,  
            encoder_attention_mask=attention_mask
        )
        return smiles_output
    
# class Mol2MSWrapperGenerator(nn.Module):
#     def __init__(self,generator_model, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         self.generator= generator_model

# # need to fix this function. 
#     def forward(self, input_ids, machine_type,collision_energy attention_mask=None):
#         # Forward pass through to convert smiles to ms_input
#             input_ids=ms_input, 
#             intensity=intensity, 
#             decoder_input_ids=ms_input,  
#             encoder_attention_mask=attention_mask
#         )
#         return smiles_output
#     # def generate(self, input_seq, start_token, device, training=False):
    #     """
    #     Autoregressive generation of a sequence based on the model's autoregressive nature.

    #     Args:
    #     - input_seq: The input sequence (e.g., SMILES or MS) to start the generation.
    #     - start_token: The initial token for autoregressive generation.
    #     - device: Device to run the model on (e.g., 'cuda' or 'cpu').
    #     - training: If True, the function assumes we're in a training phase and computes gradients.

    #     Returns:
    #     - generated_seq: The autoregressively generated sequence.
    #     """
    #     # Start with the initial input (could be a start token)
    #     generated_seq = [start_token]
        
    #     # Switch between training and inference modes
    #     if not training:
    #         self.base_model.eval()

    #     # Use torch.no_grad() only if not in training mode
    #     context_manager = torch.no_grad() if not training else torch.enable_grad()

    #     with context_manager:
    #         for _ in range(self.max_length):
    #             # Prepare the current input by appending the generated part
    #             current_input = torch.tensor([generated_seq], device=device)
                
    #             # Get the model's prediction (next token)
    #             next_token = self.base_model(current_input)
                
    #             # Use the model's prediction to generate the next token in the sequence
    #             next_token = torch.argmax(next_token, dim=-1).item()
                
    #             # Append the predicted token to the generated sequence
    #             generated_seq.append(next_token)
                
    #             # Optionally: Break the loop if an end token is generated
    #             if next_token == self.end_token:
    #                 break
        
    #     return generated_seq

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
    


class Mol2MSWrapperDiscriminator(nn.Module):
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
# i need to check one, thing,     


