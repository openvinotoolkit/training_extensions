# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Copy from mmpretrain/models/utils/swiglu_ffn.py."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from otx.algo.modules.drop import build_dropout
from otx.algo.modules.norm import build_norm_layer


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN layer.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int | None = None,
        out_dims: int | None = None,
        bias: bool = True,
        dropout_layer: dict | None = None,
        normalization: Callable[..., nn.Module] | None = None,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.out_dims = out_dims or embed_dims
        hidden_dims = feedforward_channels or embed_dims

        self.w12 = nn.Linear(self.embed_dims, 2 * hidden_dims, bias=bias)

        if normalization is not None:
            _, self.norm = build_norm_layer(normalization, hidden_dims)
        else:
            self.norm = nn.Identity()

        self.w3 = nn.Linear(hidden_dims, self.out_dims, bias=bias)
        self.gamma2 = nn.Identity()

        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor, identity: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the SwiGLUFFN module.

        Args:
            x (torch.Tensor): Input tensor.
            identity (torch.Tensor, optional): Identity tensor for residual connection. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        hidden = self.norm(hidden)
        out = self.w3(hidden)
        out = self.gamma2(out)
        out = self.dropout_layer(out)

        if self.out_dims != self.embed_dims or not self.add_identity:
            # due to the dimension inconsistence or user setting
            # not to apply residual operation
            return out

        if identity is None:
            identity = x
        return identity + out


class SwiGLUFFNFused(SwiGLUFFN):
    """SwiGLU FFN layer with fusing.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int | None = None,
        out_dims: int | None = None,
        bias: bool = True,
    ) -> None:
        out_dims = out_dims or embed_dims
        feedforward_channels = feedforward_channels or embed_dims
        feedforward_channels = (int(feedforward_channels * 2 / 3) + 7) // 8 * 8
        super().__init__(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            out_dims=out_dims,
            bias=bias,
        )
