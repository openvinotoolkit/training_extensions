"""This module contains the Decoder class for the autoencoder."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .mid import Mid
from .resnet_block import ResnetBlock


class UpBlock(nn.ModuleList):
    """UpBlock class for the decoder."""

    def __init__(self, in_channels: int, out_channels: int, is_first: bool = False):
        super().__init__(
            [
                ResnetBlock(in_channels, out_channels),
                ResnetBlock(out_channels, out_channels),
                ResnetBlock(out_channels, out_channels),
            ],
        )
        if not is_first:
            self.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UpBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self:
            if isinstance(layer, nn.Conv2d):
                bs, c, py, px = x.shape
                x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py * 2, px * 2)
            x = layer(x)
        return x


class Decoder(nn.Module):
    """Decoder class for the autoencoder."""

    def __init__(self):
        super().__init__()
        sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
        self.conv_in = nn.Conv2d(4, 512, 3, padding=1)
        self.mid = Mid(512)

        self.up_blocks = nn.ModuleList(
            [UpBlock(out_channels, in_channels, is_first=i == 0) for i, (in_channels, out_channels) in enumerate(sz)],
        )

        self.norm_out = nn.GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv_in(x)
        x = self.mid(x)

        for layer in self.up_blocks[::-1]:
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html ?
            if isinstance(layer, nn.Conv2d):
                bs, c, py, px = x.shape
                x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py * 2, px * 2)
            x = layer(x)

        return self.conv_out(F.silu(self.norm_out(x)))
