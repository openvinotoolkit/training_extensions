# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmcv/blob/98e7b3ab6088117600773245b8c6541301bfab95/mmcv/cnn/bricks/scale.py
"""Utils from mmcv."""

import torch
from torch import nn


# Come from https://github.com/open-mmlab/mmcv/blob/98e7b3ab6088117600773245b8c6541301bfab95/mmcv/cnn/bricks/scale.py
class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return x * self.scale
