# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.losses.focal_loss.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/losses/focal_loss.py
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
from otx.algo.common.losses.utils import weight_reduce_loss
from torch import nn

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
    loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
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


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        msg = f"Input labels type is not a torch.Tensor. Got {type(labels)}"
        raise TypeError(msg)

    if labels.dtype != torch.int64:
        msg = f"labels must be of the same dtype torch.int64. Got: {labels.dtype}"
        raise ValueError(msg)

    if num_classes < 1:
        msg = f"The number of classes must be bigger than one. Got: {num_classes}"
        raise ValueError(msg)
    # ipdb.set_trace()
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)
    # ipdb.set_trace()
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
    inputs: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = "none",
    eps: float | None = None,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        inputs: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C-1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> inputs = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(inputs, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(inputs, torch.Tensor):
        msg = f"inputs type is not a torch.Tensor. Got {type(inputs)}"
        raise TypeError(msg)

    if not len(inputs.shape) >= 2:
        msg = f"Invalid inputs shape, we expect BxCx*. Got: {inputs.shape}"
        raise ValueError(msg)

    if inputs.size(0) != target.size(0):
        msg = f"Expected inputs batch_size ({inputs.size(0)}) to match target batch_size ({target.size(0)})."
        raise ValueError(msg)

    n = inputs.size(0)
    out_size = (n,) + inputs.size()[2:]
    if target.size()[1:] != inputs.size()[2:]:
        msg = f"Expected target size {out_size}, got {target.size()}"
        raise ValueError(msg)

    if inputs.device != target.device:
        msg = f"inputs and target must be in the same device. Got: {inputs.device} and {target.device}"
        raise ValueError(msg)

    # compute softmax over the classes axis
    input_soft: torch.Tensor = nn.functional.softmax(inputs, dim=1)
    log_input_soft: torch.Tensor = nn.functional.log_softmax(inputs, dim=1)
    # ipdb.set_trace()
    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target,
        num_classes=inputs.shape[1],
        device=inputs.device,
        dtype=inputs.dtype,
    )

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum("bc...,bc...->b...", (target_one_hot, focal))
    # ipdb.set_trace()
    return weight_reduce_loss(loss_tmp, reduction=reduction, avg_factor=None)


class FocalLoss(nn.Module):
    """Criterion that computes Focal loss."""

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = "none", eps: float | None = None) -> None:
        r"""Criterion that computes Focal loss.

        According to :cite:`lin2018focal`, the Focal loss is computed as follows:
        .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\\gamma} \\, \text{log}(p_t)
        Where:
        - :math:`p_t` is the model's estimated probability for each class.

        Args:
        alpha: Weighting factor :math:`\alpha \\in [0, 1]`.
        gamma: Focusing parameter :math:`\\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
        output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
        will be applied, ``'mean'``: the sum of the output will be divided by
        the number of elements in the output, ``'sum'``: the output will be
        summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
        used.

        Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
        """
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = eps

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return focal_loss(inputs, target, self.alpha, self.gamma, self.reduction, self.eps)
