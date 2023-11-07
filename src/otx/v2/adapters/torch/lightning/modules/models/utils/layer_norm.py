"""Layer normalization module."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This code is a modification of the https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py

import torch
from torch import Tensor, nn


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
class LayerNorm2d(nn.Module):
    """LayerNorm2d module.

    Reference: https://github.com/facebookresearch/segment-anything
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """Initializes a LayerNorm2d module.

        Args:
            num_channels (int): The number of channels in the input tensor.
            eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of LayerNorm2d.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]
