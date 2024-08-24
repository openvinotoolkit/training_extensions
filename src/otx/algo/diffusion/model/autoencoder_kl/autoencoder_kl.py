"""AutoencoderKL module for diffusion models.

This module defines the AutoencoderKL class which includes an encoder, decoder,
and convolutional layers for quantization.
"""

import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder


class AutoencoderKL(nn.Module):
    """AutoencoderKL module."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AutoencoderKL module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        latent = self.encoder(x)
        latent = self.quant_conv(latent)
        latent = latent[:, 0:4]  # only the means
        latent = self.post_quant_conv(latent)
        return self.decoder(latent)
