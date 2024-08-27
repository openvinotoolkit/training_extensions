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

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        latent = self.encoder(x)
        return self.quant_conv(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode the latent tensor.

        Args:
            latent (torch.Tensor): Latent tensor.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        latent = self.post_quant_conv(latent)
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AutoencoderKL module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        latent = self.encode(x)
        mean = latent.chunk(2, dim=1)[0]
        return self.decode(mean)
