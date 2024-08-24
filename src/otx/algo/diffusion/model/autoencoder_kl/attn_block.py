import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class AttnBlock(nn.Module):
    """Class representing the AttnBlock module."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    # copied from AttnBlock in ldm repo
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AttnBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q, k, v = (x.reshape(b, c, h * w).transpose(1, 2) for x in (q, k, v))
        h_ = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj_out(h_)
