# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.utils.misc.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/utils/misc.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.autograd import Function

if TYPE_CHECKING:
    import torch


class SigmoidGeometricMean(Function):
    """Forward and backward function of geometric mean of two sigmoid functions.

    This implementation with analytical gradient function substitutes
    the autograd function of (x.sigmoid() * y.sigmoid()).sqrt(). The
    original implementation incurs none during gradient backprapagation
    if both x and y are very small values.
    """

    @staticmethod
    def forward(ctx: SigmoidGeometricMean, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward function of SigmoidGeometricMean."""
        x_sigmoid = x.sigmoid()
        y_sigmoid = y.sigmoid()
        z = (x_sigmoid * y_sigmoid).sqrt()
        ctx.save_for_backward(x_sigmoid, y_sigmoid, z)
        return z

    @staticmethod
    def backward(ctx: SigmoidGeometricMean, grad_output: torch.Tensor) -> tuple:
        """Backward function of SigmoidGeometricMean."""
        x_sigmoid, y_sigmoid, z = ctx.saved_tensors
        grad_x = grad_output * z * (1 - x_sigmoid) / 2
        grad_y = grad_output * z * (1 - y_sigmoid) / 2
        return grad_x, grad_y


sigmoid_geometric_mean = SigmoidGeometricMean.apply
