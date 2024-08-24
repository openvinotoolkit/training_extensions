"""This module implements a Residual Block for a neural network."""

import torch
from torch import nn


class ResBlock(nn.Module):
    """Residual block module."""

    def __init__(self, channels: int, emb_channels: int, out_channels: int):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip_connection = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else (lambda x: x)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResBlock module.

        Args:
            x (torch.Tensor): Input tensor.
            emb (torch.Tensor): Embedding tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        h = h + emb_out.reshape(*emb_out.shape, 1, 1)
        h = self.out_layers(h)
        return self.skip_connection(x) + h
