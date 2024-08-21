# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loss module to fine-tune Segment Anything Model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from otx.algo.visual_prompting.utils.postprocess import postprocess_masks

if TYPE_CHECKING:
    from torchvision import tv_tensors


class SAMCriterion(nn.Module):
    """Criterion for fine-tuning Segment Anything Model.

    TODO (sungchul): enable to get common loss modules
    """

    def __init__(self, image_size: int):
        super().__init__()
        self.image_size = image_size

    def forward(
        self,
        pred_masks: list[Tensor],
        gt_masks: list[tv_tensors.Mask],
        ious: list[Tensor],
        ori_shapes: list[Tensor],
    ) -> dict[str, Tensor]:
        """Perform loss computation."""
        loss_dice = 0.0
        loss_focal = 0.0
        loss_iou = 0.0

        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        for pred_mask, gt_mask, iou, ori_shape in zip(pred_masks, gt_masks, ious, ori_shapes):  # type: ignore[arg-type]
            post_processed_pred_mask = postprocess_masks(pred_mask, self.image_size, ori_shape)
            post_processed_pred_mask = post_processed_pred_mask.sigmoid()
            post_processed_pred_mask = post_processed_pred_mask.flatten(1)
            flatten_gt_mask = gt_mask.flatten(1).float()

            # calculate losses
            loss_dice += self.calculate_dice_loss(post_processed_pred_mask, flatten_gt_mask, num_masks)
            loss_focal += self.calculate_sigmoid_ce_focal_loss(post_processed_pred_mask, flatten_gt_mask, num_masks)
            batch_iou = self.calculate_iou(post_processed_pred_mask, flatten_gt_mask)
            loss_iou += nn.functional.mse_loss(iou, batch_iou.unsqueeze(1), reduction="sum") / num_masks

        loss = 20.0 * loss_focal + loss_dice + loss_iou

        return {"loss": loss, "loss_focal": loss_focal, "loss_dice": loss_dice, "loss_iou": loss_iou}

    def calculate_dice_loss(self, inputs: Tensor, targets: Tensor, num_masks: int) -> Tensor:
        """Compute the DICE loss, similar to generalized IOU for masks.

        TODO (sungchul): use common dice loss

        Args:
            inputs (Tensor): A tensor representing a mask.
            targets (Tensor): A tensor with the same shape as inputs. Stores the binary classification labels
                for each element in inputs (0 for the negative class and 1 for the positive class).
            num_masks (int): The number of masks present in the current batch, used for normalization.

        Returns:
            Tensor: The DICE loss.
        """
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks

    def calculate_sigmoid_ce_focal_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
        num_masks: int,
        alpha: float = 0.25,
        gamma: float = 2,
    ) -> Tensor:
        r"""Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002. # noqa: D301.

        TODO (sungchul): use common sigmoid cross entropy focal loss

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
            targets (Tensor): A tensor with the same shape as inputs. Stores the binary classification labels
                for each element in inputs (0 for the negative class and 1 for the positive class).
            num_masks (int): The number of masks present in the current batch, used for normalization.
            alpha (float, *optional*, defaults to 0.25): Weighting factor in range (0,1)
                to balance positive vs negative examples.
            gamma (float, *optional*, defaults to 2.0): Exponent of the modulating factor \\(1 - p_t\\)
                to balance easy vs hard examples.

        Returns:
            Tensor: The focal loss.
        """
        loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_masks

    def calculate_iou(self, inputs: Tensor, targets: Tensor, epsilon: float = 1e-7) -> Tensor:
        """Calculate the intersection over union (IOU) between the predicted mask and the ground truth mask.

        TODO (sungchul): use common IOU loss

        Args:
            inputs (Tensor): A tensor representing a mask.
            targets (Tensor): A tensor with the same shape as inputs. Stores the binary classification labels
                for each element in inputs (0 for the negative class and 1 for the positive class).
            epsilon (float, *optional*, defaults to 1e-7): A small value to prevent division by zero.

        Returns:
            Tensor: The IOU between the predicted mask and the ground truth mask.
        """
        pred_mask = (inputs >= 0.5).float()
        intersection = torch.sum(torch.mul(pred_mask, targets), dim=1)
        union = torch.sum(pred_mask, dim=1) + torch.sum(targets, dim=1) - intersection
        return intersection / (union + epsilon)
