"""This module contains the definition of the Mid class."""

import torch
from torch import nn

from .attn_block import AttnBlock
from .resnet_block import ResnetBlock


class Mid(nn.ModuleDict):
    """This class represents the middle layer of the autoencoder."""

    def __init__(self, block_in: int):
        super().__init__(
            {
                "block_1": ResnetBlock(block_in, block_in),
                "attn_1": AttnBlock(block_in),
                "block_2": ResnetBlock(block_in, block_in),
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Mid module."""
        for layer in self.values():
            x = layer(x)
        return x
