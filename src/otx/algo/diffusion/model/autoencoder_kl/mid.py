"""This module contains the definition of the Mid class."""

from torch import nn

from .attn_block import AttnBlock
from .resnet_block import ResnetBlock


class Mid(nn.Sequential):
    """This class represents the middle layer of the autoencoder."""

    def __init__(self, block_in: int):
        super().__init__(
            ResnetBlock(block_in, block_in),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in),
        )
