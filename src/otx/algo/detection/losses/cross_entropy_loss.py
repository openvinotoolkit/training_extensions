"""The original source code is from mmdet.mask.structures. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F  # noqa: N812
from mmengine.registry import MODELS
from torch import nn

from .utils import weight_reduce_loss


def cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    avg_factor: int | None = None,
    class_weight: list[float] | None = None,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = F.cross_entropy(pred, target, weight=class_weight, reduction="none", ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    return weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)


def _expand_onehot_labels(
    labels: torch.Tensor,
    label_weights: torch.Tensor,
    label_channels: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0), label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    avg_factor: int | None = None,
    class_weight: list[float] | None = None,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1) or (N, ).
            When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label
            will not be expanded to one-hot format.
        label (torch.Tensor): The learning label of the prediction,
            with shape (N, ).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss.
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != target.dim():
        target, weight, valid_mask = _expand_onehot_labels(target, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((target >= 0) & (target != ignore_index)).float()

        # The inplace writing method will have a mismatched broadcast
        # shape error if the weight and valid_mask dimensions
        # are inconsistent such as (B,N,1) and (B,N,C).
        weight = weight * valid_mask if weight is not None else valid_mask

    # weighted element-wise losses
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred, target.float(), pos_weight=class_weight, reduction="none")
    # do the reduction for the weighted loss
    return weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)


def mask_cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    avg_factor: int | None = None,
    class_weight: list[float] | None = None,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss

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
    if ignore_index is not None:
        msg = "ignore_index is not supported in mask cross entropy loss"
        raise ValueError(msg)
    if reduction != "mean" or avg_factor is not None:
        msg = "avg_factor is not supported in mask cross entropy loss"
        raise ValueError(msg)
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, weight].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, weight=class_weight, reduction="mean")[None]


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss."""

    def __init__(
        self,
        use_sigmoid: bool = False,
        use_mask: bool = False,
        reduction: str = "mean",
        class_weight: list[float] | None = None,
        ignore_index: int | None = None,
        loss_weight: float = 1.0,
        avg_non_ignore: bool = False,
    ):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super().__init__()
        if use_sigmoid and use_mask:
            msg = "``use_sigmoid`` and ``use_mask`` cannot be True at the same time"
            raise ValueError(msg)

        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if (ignore_index is not None) and not self.avg_non_ignore and self.reduction == "mean":
            warnings.warn(
                "Default ``avg_non_ignore`` is False, if you would like to "
                "ignore the certain label and average loss over non-ignore "
                "labels, which is the same with PyTorch official "
                "cross_entropy, set ``avg_non_ignore=True``.",
                stacklevel=2,
            )

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def extra_repr(self) -> str:
        """Extra repr."""
        return f"avg_non_ignore={self.avg_non_ignore}"

    def forward(
        self,
        cls_score: torch.Tensor,
        label: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
        ignore_index: int | None = None,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.

        Returns:
            torch.Tensor: The calculated loss.
        """
        if reduction_override not in (None, "none", "mean", "sum"):
            msg = f"Invalid value for reduction_override: {reduction_override}"
            raise ValueError(msg)
        reduction = reduction_override if reduction_override else self.reduction
        if ignore_index is None:
            ignore_index = self.ignore_index

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
        )
