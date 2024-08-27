"""This module contains the ClipTextEmbeddings class, which is an embedding module for CLIP text inputs."""

import torch
from torch import nn


class TextEmbeddings(nn.Module):
    """Embedding module for CLIP text inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(49408, 768)
        self.position_embedding = nn.Embedding(77, 768)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ClipTextEmbeddings module.

        Args:
            input_ids (torch.Tensor): The input tensor representing the tokenized text.
            position_ids (torch.Tensor): The input tensor representing the position of each token.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)
