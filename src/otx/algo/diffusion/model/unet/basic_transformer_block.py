"""A module that defines the BasicTransformerBlock class."""

from __future__ import annotations

import torch
from torch import nn

from .cross_attention import CrossAttention
from .feed_forward import FeedForward


class BasicTransformerBlock(nn.Module):
    """A basic transformer block implementation."""

    def __init__(self, dim: int, ctx_dim: int, n_heads: int, d_head: int):
        super().__init__()
        self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(dim, ctx_dim, n_heads, d_head)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the BasicTransformerBlock.

        Args:
            x (torch.Tensor): The input tensor.
            ctx (torch.Tensor, optional): The context tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), ctx=ctx)
        return x + self.ff(self.norm3(x))
