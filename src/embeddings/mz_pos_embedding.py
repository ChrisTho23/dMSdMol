import math

import torch
from jaxtyping import Float
from torch import Tensor, nn


class MZPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, freq_scale: float = 1.0, normalize: bool = False):
        """
        Args:
            d_model (int): Dimension of the positional encoding vectors.
            freq_scale (float): Scaling factor for the frequency components.
            normalize (bool): Whether to normalize positions to have zero mean.
        """
        super(MZPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.freq_scale = freq_scale
        self.normalize = normalize

        # Frequency components for encoding
        freq = freq_scale * torch.exp(-2.0 * torch.arange(d_model // 2) * (math.log(1e4) / d_model))
        self.register_buffer('freq', freq)

        # Phase shifts for cosine and sine (π/2 shift)
        _sin2cos_phase_shift = torch.pi / 2.0
        cos_shifts = _sin2cos_phase_shift * (torch.arange(d_model) % 2)
        self.register_buffer('cos_shifts', cos_shifts)
        
    def forward(self, continuous_values: Float[Tensor, "batch seq"]) -> Float[Tensor, "batch seq d_model"]:
        """
        Args:
            continuous_values (Tensor): Continuous input tensor of shape (batch_size, seq).
        
        Returns:
            Tensor: Positional encoding of shape (batch_size, seq, d_model).
        """
        batch_size, seq = continuous_values.shape
        
        # Optionally normalize the continuous values (to focus on relational distances)
        if self.normalize:
            continuous_values = continuous_values - torch.mean(continuous_values, dim=1, keepdim=True)
        
        # Expand continuous values to shape (batch_size, seq, 1) to prepare for multiplication
        continuous_values = continuous_values.unsqueeze(-1)

        # Compute the product of positions and frequencies for encoding
        product = continuous_values * self.freq
        
        # Apply sine and cosine functions to get the positional embeddings
        pos_emb = torch.sin(product + self.cos_shifts)
        
        return pos_emb
