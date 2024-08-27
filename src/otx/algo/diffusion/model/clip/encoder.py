"""Module representing the CLIP encoder model."""

from __future__ import annotations

import torch
from torch import nn

from .encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """Class representing the CLIP encoder model."""

    def __init__(self, layer_count: int = 12):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(layer_count)])

    def forward(
        self,
        x: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the CLIP encoder model.

        Args:
            x (torch.Tensor): Input tensor.
            causal_attention_mask (torch.Tensor): Causal attention mask tensor.
            ret_layer_idx (int | None, optional): Index of the layer to return. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        # the indexing of layers is NOT off by 1, the original code considers the "input" as the first hidden state
        for layer in self.layers:
            x = layer(x, causal_attention_mask)
        return x
