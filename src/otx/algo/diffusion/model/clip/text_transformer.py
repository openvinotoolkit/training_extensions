"""This module provides the implementation of the ClipTextTransformer class."""

from __future__ import annotations

import torch
from torch import nn

from .encoder import Encoder
from .text_embeddings import TextEmbeddings


class TextTransformer(nn.Module):
    """Transformer model for ClipText."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = TextEmbeddings()
        self.encoder = Encoder()
        self.final_layer_norm = nn.LayerNorm(768)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ClipTextTransformer."""
        x = self.embeddings(input_ids, torch.arange(input_ids.shape[1]).reshape(1, -1).to(input_ids.device))
        x = self.encoder(x, torch.full((1, 1, 77, 77), float("-inf")).triu(1).to(x.device))
        return self.final_layer_norm(x)
