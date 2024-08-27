"""This module contains the implementation of the SpatialTransformer class."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .basic_transformer_block import BasicTransformerBlock


class SpatialTransformer(nn.Module):
    """Spatial Transformer module."""

    def __init__(
        self,
        channels: int,
        n_heads: int,
        d_head: int,
        ctx_dim: int | list[int],
        use_linear: bool,
        depth: int = 1,
    ):
        super().__init__()
        if isinstance(ctx_dim, int):
            ctx_dim = [ctx_dim] * depth
        else:
            if not isinstance(ctx_dim, list):
                msg = "ctx_dim must be a list"
                raise TypeError(msg)
            if depth != len(ctx_dim):
                msg = "depth must be equal to the length of ctx_dim"
                raise ValueError(msg)
        self.norm = nn.GroupNorm(32, channels)
        if channels != n_heads * d_head:
            msg = "channels must be equal to n_heads * d_head"
            raise ValueError(msg)
        self.proj_in = nn.Linear(channels, channels) if use_linear else nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, ctx_dim[d], n_heads, d_head) for d in range(depth)],
        )
        self.proj_out = nn.Linear(channels, channels) if use_linear else nn.Conv2d(channels, channels, 1)
        self.use_linear = use_linear

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the SpatialTransformer module.

        Args:
            x (torch.Tensor): Input tensor.
            ctx (torch.Tensor, optional): Context tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        ops: list[Callable[[torch.Tensor], torch.Tensor]] = [
            (lambda z: z.reshape(b, c, h * w).permute(0, 2, 1)),
            self.proj_in,
        ]
        if not self.use_linear:
            ops = list(reversed(ops))
        for op in ops:
            x = op(x)
        for block in self.transformer_blocks:
            x = block(x, ctx=ctx)
        ops = [(lambda z: self.proj_out(z)), (lambda z: z.permute(0, 2, 1).reshape(b, c, h, w))]
        if not self.use_linear:
            ops = list(reversed(ops))
        for op in ops:
            x = op(x)
        return x + x_in
