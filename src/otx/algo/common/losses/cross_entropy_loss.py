# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.losses.cross_entropy.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/losses/cross_entropy.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .utils import weight_reduce_loss


def cross_entropy(
    pred: Tensor,
    label: Tensor,
    weight: Tensor | None = None,
    reduction: str = "mean",
    avg_factor: int | None = None,
    class_weight: list[float] | None = None,
    ignore_index: int = -100,
    avg_non_ignore: bool = False,
) -> Tensor:
    """Calculate the CrossEntropy loss.

    Args:
        pred (Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (Tensor): The learning label of the prediction.
        weight (Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int): The label index to be ignored.
            Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        Tensor: The calculated loss
    """
    loss = nn.functional.cross_entropy(pred, label, weight=class_weight, reduction="none", ignore_index=ignore_index)

    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660
    if (avg_factor is None) and avg_non_ignore and reduction == "mean":
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    return weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)


def _expand_onehot_labels(
    labels: Tensor,
    label_weights: Tensor,
    label_channels: int,
    ignore_index: int,
) -> tuple[Tensor, ...]:
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0), label_channels).float()
    bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
    bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(
    pred: Tensor,
    label: Tensor,
    weight: Tensor | None = None,
    reduction: str = "mean",
    avg_factor: int | None = None,
    class_weight: list[float] | None = None,
    ignore_index: int = -100,
    avg_non_ignore: bool = False,
) -> Tensor:
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (Tensor): The prediction with shape (N, 1) or (N, ).
            When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label
            will not be expanded to one-hot format.
        label (Tensor): The learning label of the prediction,
            with shape (N, ).
        weight (Tensor, None): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int): The label index to be ignored.
            Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        Tensor: The calculated loss.
    """
    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(label, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        # The inplace writing method will have a mismatched broadcast
        # shape error if the weight and valid_mask dimensions
        # are inconsistent such as (B,N,1) and (B,N,C).
        weight = weight * valid_mask if weight is not None else valid_mask

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == "mean":
        avg_factor = valid_mask.sum().item()

    # weighted element-wise losses
    weight = weight.float()
    loss = nn.functional.binary_cross_entropy_with_logits(
        pred,
        label.float(),
        pos_weight=class_weight,
        reduction="none",
    )
    # do the reduction for the weighted loss
    return weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)


def mask_cross_entropy(
    pred: Tensor,
    target: Tensor,
    label: Tensor,
    class_weight: list[float] | None = None,
    **kwargs,  # noqa: ARG001
) -> Tensor:
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (Tensor): The learning label of the prediction.
        label (Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        class_weight (list[float], None): The weight for each class.

    Returns:
        Tensor: The calculated loss

    Example:
        >>> N, C = 3, 11
        >>> H, W = 2, 2
        >>> pred = torch.randn(N, C, H, W) * 1000
        >>> target = torch.rand(N, H, W)
        >>> label = torch.randint(0, C, size=(N,))
        >>> reduction = 'mean'
        >>> avg_factor = None
        >>> class_weights = None
        >>> loss = mask_cross_entropy(pred, target, label, reduction,
        >>>                           avg_factor, class_weights)
        >>> assert loss.shape == (1,)
    """
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return nn.functional.binary_cross_entropy_with_logits(
        pred_slice,
        target,
        weight=class_weight,
        reduction="mean",
    )[None]


class CrossEntropyLoss(nn.Module):
    """Base Cross Entropy Loss implementation from mmdet."""

    def __init__(
        self,
        use_sigmoid: bool = False,
        use_mask: bool = False,
        reduction: str = "mean",
        class_weight: list[float] | None = None,
        loss_weight: float = 1.0,
        avg_non_ignore: bool = False,
    ):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.avg_non_ignore = avg_non_ignore

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy  # type: ignore[assignment]
        else:
            self.cls_criterion = cross_entropy

    def extra_repr(self) -> str:
        """Extra repr."""
        return f"avg_non_ignore={self.avg_non_ignore}"

    def forward(
        self,
        cls_score: Tensor,
        label: Tensor,
        weight: Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
        ignore_index: int = -100,
        **kwargs,
    ) -> Tensor:
        """Forward function.

        Args:
            cls_score (Tensor): The prediction.
            label (Tensor): The learning label of the prediction.
            weight (Tensor, None): Sample-wise loss weight.
            avg_factor (int, None): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, None): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int): The label index to be ignored.
                Default: -100.

        Returns:
            Tensor: The calculated loss.
        """
        reduction = reduction_override if reduction_override else self.reduction

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        return self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs,
        )
