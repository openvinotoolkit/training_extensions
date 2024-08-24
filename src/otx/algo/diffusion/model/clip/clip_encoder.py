"""Module representing the CLIP encoder model."""

from __future__ import annotations

import torch
from torch import nn

from .clip_encoder_layer import ClipEncoderLayer


class ClipEncoder(nn.Module):
    """Class representing the CLIP encoder model."""

    def __init__(self, layer_count: int = 12):
        super().__init__()
        self.layers = nn.ModuleList([ClipEncoderLayer() for _ in range(layer_count)])

    def forward(
        self,
        x: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        ret_layer_idx: int | None = None,
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
        layers = self.layers if ret_layer_idx is None else self.layers[:ret_layer_idx]
        for layer in layers:
            x = layer(x, causal_attention_mask)
        return x
