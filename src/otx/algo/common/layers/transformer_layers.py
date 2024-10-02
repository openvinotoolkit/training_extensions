# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of common transformer layers."""

from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import nn


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.GELU,
        normalize_before: bool = False,
        batch_first: bool = True,
        key_mask: bool = False,
    ) -> None:
        super().__init__()
        self.normalize_before = normalize_before
        self.key_mask = key_mask

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos_embed: torch.Tensor | None) -> torch.Tensor:
        """Attach position embeddings to the tensor."""
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward the transformer encoder layer.

        Args:
            src (torch.Tensor): The input tensor.
            src_mask (torch.Tensor | None, optional): The mask tensor. Defaults to None.
            pos_embed (torch.Tensor | None, optional): The position embedding tensor. Defaults to None.
        """
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        if self.key_mask:
            src = self.self_attn(q, k, value=src, key_padding_mask=src_mask)[0]
        else:
            src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    """TransformerEncoder."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None) -> None:
        """Initialize the TransformerEncoder.

        Args:
            encoder_layer (nn.Module): The encoder layer module.
            num_layers (int): The number of layers.
            norm (nn.Module | None, optional): The normalization module. Defaults to None.
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward the transformer encoder.

        Args:
            src (torch.Tensor): The input tensor.
            src_mask (torch.Tensor | None, optional): The mask tensor. Defaults to None.
            pos_embed (torch.Tensor | None, optional): The position embedding tensor. Defaults to None.
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output
