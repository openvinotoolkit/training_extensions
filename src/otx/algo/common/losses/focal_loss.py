# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.losses.focal_loss.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/losses/focal_loss.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional
from otx.algo.common.losses.utils import weight_reduce_loss

if TYPE_CHECKING:
    from torch import Tensor


# This method is only for debugging
def py_sigmoid_focal_loss(
    pred: Tensor,
    target: Tensor,
    weight: None | Tensor = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
    avg_factor: int | None = None,
) -> torch.Tensor:
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                if weight.numel() != loss.numel():
                    msg = "The number of elements in weight should be equal to the number of elements in loss."
                    raise ValueError(msg)
                weight = weight.view(loss.size(0), -1)
        if weight.ndim != loss.ndim:
            msg = "The number of dimensions in weight should be equal to the number of dimensions in loss."
            raise ValueError(msg)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)
