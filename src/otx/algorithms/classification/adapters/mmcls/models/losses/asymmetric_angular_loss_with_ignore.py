"""Module for defining AsymmetricAngularLossWithIgnore."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import LOSSES
from mmcls.models.losses.utils import weight_reduce_loss
from torch import nn


def asymmetric_angular_loss_with_ignore(
    pred,
    target,
    valid_label_mask=None,
    weight=None,
    gamma_pos=0.0,
    gamma_neg=1.0,
    clip=0.05,
    k=0.8,
    reduction="mean",
    avg_factor=None,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Asymmetric angular loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, *).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, *).
        valid_label_mask (torch.Tensor, optional): Label mask for consideration
            of ignored label.
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Dafaults to None.
        gamma_pos (float): positive focusing parameter. Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We usually set
            gamma_neg > gamma_pos. Defaults to 1.0.
        k (float): positive balance parameter. Defaults to 0.8.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
             is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == target.shape, "pred and target should be in the same shape."

    eps = 1e-8
    target = target.type_as(pred)
    anti_target = 1 - target

    xs_pos = torch.sigmoid(pred)
    xs_neg = torch.sigmoid(-pred)

    if clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1)

    asymmetric_focus = gamma_pos > 0 or gamma_neg > 0
    if asymmetric_focus:
        pos_target0 = xs_neg * target
        pos_target1 = xs_pos * anti_target
        pos_target = pos_target0 + pos_target1
        one_sided_gamma = gamma_pos * target + gamma_neg * anti_target
        one_sided_w = torch.pow(pos_target, one_sided_gamma)

    loss = -k * target * torch.log(xs_pos.clamp(min=eps)) - (1 - k) * anti_target * torch.log(xs_neg.clamp(min=eps))

    if asymmetric_focus:
        loss *= one_sided_w

    if valid_label_mask is not None:
        loss = loss * valid_label_mask

    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    if reduction != "mean":
        avg_factor = None
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class AsymmetricAngularLossWithIgnore(nn.Module):
    """Asymmetric angular loss.

    Args:
        gamma_pos (float): positive focusing parameter.
            Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We
            usually set gamma_neg > gamma_pos. Defaults to 1.0.
        k (float): positive balance parameter. Defaults to 0.8.
        clip (float): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss into
            a scalar.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, gamma_pos=0.0, gamma_neg=1.0, k=0.8, clip=0.05, reduction="mean", loss_weight=1.0):
        """Init fuction of AsymmetricAngularLossWithIgnore class."""
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.k = k
        self.clip = clip
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, valid_label_mask=None, weight=None, avg_factor=None, reduction_override=None):
        """Asymmetric angular loss."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * asymmetric_angular_loss_with_ignore(
            pred,
            target,
            valid_label_mask,
            weight,
            gamma_pos=self.gamma_pos,
            gamma_neg=self.gamma_neg,
            k=self.k,
            clip=self.clip,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss_cls
