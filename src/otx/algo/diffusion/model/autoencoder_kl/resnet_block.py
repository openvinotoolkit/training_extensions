"""This module contains the definition of the ResnetBlock class, which is a residual block for a ResNet-based model."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class ResnetBlock(nn.Module):
    """Residual block for a ResNet-based model."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the ResnetBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the ResnetBlock.
        """
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.nin_shortcut(x) + h
