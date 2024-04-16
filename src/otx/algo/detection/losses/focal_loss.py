"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

# TODO(Eugene): replace mmcv.sigmoid_focal_loss with torchvision
# https://github.com/openvinotoolkit/training_extensions/pull/3281
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from mmengine.registry import MODELS
from torch import nn

from .utils import weight_reduce_loss

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
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
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


def py_focal_loss_with_prob(
    pred: Tensor,
    target: Tensor,
    weight: None | Tensor = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
    avg_factor: int | None = None,
) -> torch.Tensor:
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`.

    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
            The target shape support (N,C) or (N,), (N,C) means one-hot form.
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
    if pred.dim() != target.dim():
        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(pred, target, reduction="none") * focal_weight
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


def sigmoid_focal_loss(
    pred: Tensor,
    target: Tensor,
    weight: None | Tensor = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
    avg_factor: int | None = None,
) -> torch.Tensor:
    r"""A wrapper of cuda version `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma, alpha, None, "none")
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


@MODELS.register_module()
class FocalLoss(nn.Module):
    """Focal Loss."""

    def __init__(
        self,
        use_sigmoid: int = True,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        activated: bool = False,
    ):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super().__init__()
        if use_sigmoid is False:
            msg = "Only sigmoid focal loss supported now."
            raise NotImplementedError(msg)
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        if reduction_override not in (None, "none", "mean", "sum"):
            msg = f"reduction_override must be one of [None, 'none', 'mean', 'sum'], got {reduction_override}"
            raise ValueError(msg)
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            elif pred.dim() == target.dim():
                # this means that target is already in One-Hot form.
                calculate_loss_func = py_sigmoid_focal_loss
            elif torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        else:
            raise NotImplementedError
        return loss_cls
