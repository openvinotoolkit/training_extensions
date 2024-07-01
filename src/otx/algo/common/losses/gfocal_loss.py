# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.losses.gfocal_loss.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/losses/gfocal_loss.py
"""

from __future__ import annotations

from functools import partial

import torch.nn.functional as F  # noqa: N812
from otx.algo.common.losses.utils import weighted_loss
from torch import Tensor, nn


@weighted_loss
def quality_focal_loss_tensor_target(
    pred: Tensor,
    target: Tensor,
    beta: float = 2.0,
    activated: bool = False,
) -> Tensor:
    """QualityFocal Loss <https://arxiv.org/abs/2008.13367>.

    Args:
        pred (Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        activated (bool): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
    """
    # pred and target should be of the same size
    if pred.size() != target.size():
        msg = "The size of the prediction and target should be the same."
        raise ValueError(msg)
    if activated:
        pred_sigmoid = pred
        loss_function = F.binary_cross_entropy
    else:
        pred_sigmoid = pred.sigmoid()
        loss_function = F.binary_cross_entropy_with_logits

    scale_factor = pred_sigmoid
    target = target.type_as(pred)

    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = loss_function(pred, zerolabel, reduction="none") * scale_factor.pow(beta)

    pos = target != 0
    scale_factor = target[pos] - pred_sigmoid[pos]
    loss[pos] = loss_function(pred[pos], target[pos], reduction="none") * scale_factor.abs().pow(beta)

    return loss.sum(dim=1, keepdim=False)


@weighted_loss
def quality_focal_loss(pred: Tensor, target: Tensor, beta: float = 2.0) -> Tensor:
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    if len(target) != 2:
        msg = "The length of target should be 2."
        raise ValueError(msg)
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(pred, zerolabel, reduction="none") * scale_factor.pow(
        beta,
    )

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label],
        score[pos],
        reduction="none",
    ) * scale_factor.abs().pow(beta)

    return loss.sum(dim=1, keepdim=False)


@weighted_loss
def quality_focal_loss_with_prob(pred: Tensor, target: Tensor, beta: float = 2.0) -> Tensor:
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    if len(target) != 2:
        msg = "The length of target should be 2."
        raise ValueError(msg)
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy(pred, zerolabel, reduction="none") * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy(
        pred[pos, pos_label],
        score[pos],
        reduction="none",
    ) * scale_factor.abs().pow(beta)

    return loss.sum(dim=1, keepdim=False)


class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
        activated (bool, optional): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
    """

    def __init__(
        self,
        use_sigmoid: bool = True,
        beta: float = 2.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        activated: bool = False,
    ):
        super().__init__()
        if not use_sigmoid:
            msg = "Only sigmoid in QFL supported now."
            raise NotImplementedError(msg)
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (Union(tuple([Tensor]),Tensor)): The type is
                tuple, it should be included Target category label with
                shape (N,) and target quality label with shape (N,).The type
                is Tensor, the target should be one-hot form with
                soft weights.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        if reduction_override not in (None, "none", "mean", "sum"):
            msg = "Invalid reduction method."
            raise ValueError(msg)
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            calculate_loss_func = quality_focal_loss_with_prob if self.activated else quality_focal_loss
            if isinstance(target, Tensor):
                # the target shape with (N,C) or (N,C,...), which means
                # the target is one-hot form with soft weights.
                calculate_loss_func = partial(quality_focal_loss_tensor_target, activated=self.activated)

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        else:
            raise NotImplementedError
        return loss_cls
