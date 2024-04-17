# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Cross entropy loss for ignored mode in class-incremental learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss


class CrossEntropyLossWithIgnore(CrossEntropyLoss):
    """CrossEntropyLossWithIgnore with Ignore Mode Support for Class Incremental Learning.

    When new classes are added through continual training cycles, images from previous cycles
    may become partially annotated if they are not revisited.
    To prevent the model from predicting these new classes for such images,
    CrossEntropyLossWithIgnore can be used to ignore the unseen classes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_name = "loss_ce_ignore"

    def forward(
        self,
        cls_score: torch.Tensor,
        label: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str = "mean",
        valid_label_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward.

        Args:
            cls_score (torch.Tensor, optional): The prediction with shape (N, 1).
            label (torch.Tensor, optional): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
                Default: None.
            class_weight (list[float], optional): The weight for each class.
                Default: None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str, optional): The method used to reduce the loss.
                Options are 'none', 'mean' and 'sum'. Default: 'mean'.
            valid_label_mask (torch.Tensor, optional): The valid labels with
                shape (N, num_classes).
                If the value in the valid_label_mask is 0, mask label of the
                the mask label of the class corresponding to its index will be
                ignored like ignore_index.
            **kwargs (Any): Additional keyword arguments.
        """
        if valid_label_mask is None:
            return super().forward(cls_score, label)
        reduction = reduction_override if reduction_override else self.reduction
        batch_size = label.shape[0]
        for i in range(batch_size):
            invalid_labels = (valid_label_mask[i] == 0).nonzero(as_tuple=False)

            for inv_l in invalid_labels:
                cls_score = torch.cat((cls_score[:, :inv_l], cls_score[:, inv_l + 1 :]), dim=1)

        losses = F.cross_entropy(cls_score, label, reduction="none", ignore_index=self.ignore_index)

        if weight is not None:
            weight = weight.float()
        return weight_reduce_loss(losses, weight=weight, reduction=reduction, avg_factor=avg_factor)

    @property
    def loss_name(self) -> str:
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


def weight_reduce_loss(loss,
                       weight=None,
                       reduction='mean',
                       avg_factor=None) -> torch.Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def reduce_loss(loss, reduction) -> torch.Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
