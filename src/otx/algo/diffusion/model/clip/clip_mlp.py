"""This module contains the ClipMlp class, a multi-layer perceptron for CLIP."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class ClipMlp(nn.Module):
    """Multi-layer perceptron for CLIP."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ClipMlp module.

        Args:
            h (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        h = self.fc1(h)
        h = F.gelu(h)
        return self.fc2(h)
