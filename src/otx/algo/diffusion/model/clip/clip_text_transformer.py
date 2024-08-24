"""This module provides the implementation of the ClipTextTransformer class."""

from __future__ import annotations

import torch
from torch import nn

from .clip_encoder import ClipEncoder
from .clip_text_embeddings import ClipTextEmbeddings


class ClipTextTransformer(nn.Module):
    """Transformer model for ClipText."""

    def __init__(self, ret_layer_idx: int | None = None):
        super().__init__()
        self.embeddings = ClipTextEmbeddings()
        self.encoder = ClipEncoder()
        self.final_layer_norm = nn.LayerNorm(768)
        self.ret_layer_idx = ret_layer_idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ClipTextTransformer."""
        x = self.embeddings(input_ids, torch.arange(input_ids.shape[1]).reshape(1, -1).to(input_ids.device))
        x = self.encoder(x, torch.full((1, 1, 77, 77), float("-inf")).triu(1).to(x.device), self.ret_layer_idx)
        return self.final_layer_norm(x) if (self.ret_layer_idx is None) else x
