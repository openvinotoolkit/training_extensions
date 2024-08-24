"""This module implements the CrossAttention class for performing cross-attention operations."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class CrossAttention(nn.Module):
    """Class implementing the CrossAttention module."""

    def __init__(self, query_dim: int, ctx_dim: int, n_heads: int, d_head: int):
        super().__init__()
        self.to_q = nn.Linear(query_dim, n_heads * d_head, bias=False)
        self.to_k = nn.Linear(ctx_dim, n_heads * d_head, bias=False)
        self.to_v = nn.Linear(ctx_dim, n_heads * d_head, bias=False)
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = nn.Linear(n_heads * d_head, query_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        """Perform forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): The input tensor.
            ctx (torch.Tensor, optional): The context tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx = x if ctx is None else ctx
        q, k, v = self.to_q(x), self.to_k(ctx), self.to_v(ctx)
        q, k, v = (y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(1, 2) for y in (q, k, v))
        attention = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        h_ = attention.reshape(x.shape[0], -1, self.num_heads * self.head_size)
        return self.to_out(h_)
