# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for defining AsymmetricAngularLossWithIgnore."""

from __future__ import annotations

import torch
from torch import nn


def reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = nn.functional._Reduction.get_enum(reduction)  # noqa: SLF001
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()
    return loss.sum()


def weight_reduce_loss(
    loss: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> torch.Tensor:
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
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        return reduce_loss(loss, reduction)
    # if reduction is mean, then average the loss by avg_factor
    if reduction == "mean":
        return loss.sum() / avg_factor
    # if reduction is 'none', then do nothing, otherwise raise an error
    if reduction == "none":
        return loss
    msg = 'avg_factor can not be used with reduction="sum"'
    raise ValueError(msg)


def asymmetric_angular_loss_with_ignore(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_label_mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    gamma_pos: float = 0.0,
    gamma_neg: float = 1.0,
    clip: float = 0.05,
    k: float = 0.8,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> torch.Tensor:
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
        avg_factor (float, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """
    if pred.shape != target.shape:
        msg = "pred and target should be in the same shape."
        raise ValueError(msg)

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
        if weight.dim() != 1:
            raise ValueError
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    if reduction != "mean":
        avg_factor = None
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


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

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 1.0,
        k: float = 0.8,
        clip: float = 0.05,
        reduction: str = "sum",
        loss_weight: float = 1.0,
    ):
        """Init fuction of AsymmetricAngularLossWithIgnore class."""
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.k = k
        self.clip = clip
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_label_mask: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
    ) -> torch.Tensor:
        """Asymmetric angular loss."""
        if reduction_override not in (None, "none", "mean", "sum"):
            msg = f"reduction_override should be none / mean / sum / None, {reduction_override}"
            raise ValueError(msg)
        reduction = reduction_override if reduction_override else self.reduction

        return self.loss_weight * asymmetric_angular_loss_with_ignore(
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
