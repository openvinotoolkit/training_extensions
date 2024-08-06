# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation of ScaleNorm."""
import torch
from torch import nn


class ScaleNorm(nn.Module):
    """Scale Norm.

    Args:
        dim (int): The dimension of the scale vector.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.

    Reference:
        `Transformers without Tears: Improving the Normalization
        of Self-Attention <https://arxiv.org/abs/1910.05895>`_
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The tensor after applying scale norm.
        """
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        norm = norm * self.scale
        return x / norm.clamp(min=self.eps) * self.g
