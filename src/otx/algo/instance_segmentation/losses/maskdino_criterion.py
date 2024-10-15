# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""MaskDINO criterion."""
from __future__ import annotations

from typing import Callable

import torch
import torch.distributed
import torch.nn.functional as f
from torch import Tensor, nn
from torchvision.ops import box_convert

from otx.algo.common.layers.hungarian_matcher import HungarianMatcher
from otx.algo.common.losses import GIoULoss, L1Loss
from otx.algo.instance_segmentation.utils.utils import get_uncertain_point_coords_with_randomness, point_sample


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2,
) -> Tensor:
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape. The predictions for each example.
        targets (torch.Tesnor): A float tensor with the same shape as inputs. Stores the binary classification label.
        num_boxes (float): Number of boxes.
        alpha (float): Weighting factor in the loss function. Default to 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Default to 2.

    Returns:
        torch.Tensor: Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = f.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(
    inputs: Tensor,
    targets: Tensor,
    num_masks: float,
) -> Tensor:
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape. The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs.
        num_masks (float): Number of masks.

    Returns:
        Tensor: Loss tensor
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(
    inputs: Tensor,
    targets: Tensor,
    num_masks: float,
) -> Tensor:
    """Compute the sigmoid cross entropy loss.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape. The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary classification label.

    Returns:
        Tensor: Loss tensor
    """
    loss = f.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


def calculate_uncertainty(logits: Tensor) -> Tensor:
    """Calculate uncertainty as L1 dist between 0.0 and logit prediction.

    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.

    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] != 1:
        msg = "The input tensor must be of shape (R, 1, ...)"
        raise ValueError(msg)
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def select_masks(tgt_idx: Tensor, mask_labels: Tensor) -> Tensor:
    """Select masks from mask_labels based on tgt_idx.

    Args:
        tgt_idx (Tensor): target index
        mask_labels (Tensor): mask labels

    Returns:
        Tensor: selected target masks.
    """
    batch_size = torch.max(tgt_idx[0]) + 1
    gt_masks = [mask_labels[b][tgt_idx[1][tgt_idx[0] == b]] for b in range(batch_size)]
    return torch.cat(gt_masks, dim=0)


class MaskDINOCriterion(nn.Module):
    """This class computes the loss for MaskDINO.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)

    Args:
        num_classes (int): number of object categories, omitting the special no-object category
        matcher (HungarianMatcher): module that computes the matching between ground truth and predicted boxes
        weight_dict (dict, optional): dict containing key names of the losses and as values their relative weight.
        eos_coef (float): relative classification weight applied to the no-object category
        losses (list): list of all the losses to be applied
        num_points (int): number of points to sample for pointwise mask loss
        oversample_ratio (float): ratio of oversampling for pointwise mask loss
        importance_sample_ratio (float): ratio of importance sampling for pointwise mask loss
        dec_layers (int): number of decoder layers
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float] | None = None,
        eos_coef: float = 0.1,
        num_points: int = 112 * 112,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        dec_layers: int = 9,
    ) -> None:
        super().__init__()
        if weight_dict is None:
            dec_layers = 9
            weight_dict = {
                "loss_ce": 4.0,
                "loss_dice": 5.0,
                "loss_mask": 5.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            }
            weight_dict.update({k + "_interm": v for k, v in weight_dict.items()})

            # denoising training
            weight_dict.update({k + "_dn": v for k, v in weight_dict.items()})

            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        loss_bbox_weight = weight_dict["loss_bbox"] if "loss_bbox" in weight_dict else 1.0
        loss_giou_weight = weight_dict["loss_giou"] if "loss_giou" in weight_dict else 1.0
        self.lossl1 = L1Loss(loss_weight=loss_bbox_weight)
        self.giou = GIoULoss(loss_weight=loss_giou_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25

    def loss_labels(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> dict[str, Tensor]:
        """Compute the losses related to the labels: the classification loss.

        Args:
            outputs (dict[str, Tensor]): The model outputs.
            targets (list[dict[str, Tensor]]): The targets.
            indices (list[tuple[Tensor, Tensor]]): The indices of the matched labels.
            num_boxes (float): Number of boxes.

        Returns:
            dict[str, Tensor]: The computed losses.
        """
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices, strict=True)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * src_logits.shape[1]
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> dict[str, Tensor]:
        """Compute the losses related to the bounding boxes: the L1 loss and the GIoU loss.

        Args:
            outputs (dict[str, Tensor]): The model outputs.
            targets (list[dict[str, Tensor]]): The targets.
            indices (list[tuple[Tensor, Tensor]]): The indices of the matched boxes.
            num_boxes (float): Number of boxes.

        Returns:
            dict[str, Tensor]: The computed losses.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices, strict=True)], dim=0)

        losses = {}

        loss_bbox = self.lossl1(src_boxes, target_boxes, avg_factor=num_boxes)
        loss_giou = self.giou(
            box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            avg_factor=num_boxes,
        )
        losses["loss_giou"] = loss_giou
        losses["loss_bbox"] = loss_bbox

        return losses

    def loss_masks(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_masks: float,
    ) -> dict[str, Tensor]:
        """Compute the losses related to the masks: the focal loss and dice loss.

        Args:
            outputs (dict[str, Tensor]): The model outputs.
            targets (list[dict[str, Tensor]]): The targets.
            indices (list[tuple[Tensor, Tensor]]): The matched indices for masks.
            num_masks (float): Number of masks.

        Returns:
            dict[str, Tensor]: The computed losses.
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        pred_masks = outputs["pred_masks"]
        pred_masks = pred_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks = select_masks(tgt_idx, masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                pred_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks.to(pred_masks),
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            pred_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    def prep_for_dn(self, mask_dict: dict) -> tuple[dict[str, Tensor], int, int, int]:
        """Prepare denoised output for denoise loss computation.

        Args:
            mask_dict (dict): denoise output information.

        Raises:
            ValueError: pad_size must be divisible by scalar.

        Returns:
            tuple[dict[str, Tensor], int, int, int]: output_known_lbs_bboxes, num_tgt, single_pad, scalar.
                - output_known_lbs_bboxes: output known labels and bboxes.
                - num_tgt: number of targets.
                - single_pad: single pad size.
                - scalar: scalar value.
        """
        output_known_lbs_bboxes = mask_dict["output_known_lbs_bboxes"]

        known_indice = mask_dict["known_indice"]
        scalar, pad_size = mask_dict["scalar"], mask_dict["pad_size"]
        if pad_size % scalar != 0:
            msg = "pad_size must be divisible by scalar."
            raise ValueError(msg)
        single_pad = pad_size // scalar

        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes, num_tgt, single_pad, scalar

    def _get_src_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Permute predictions following indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Permute targets following indices."""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @property
    def _available_losses(self) -> tuple[Callable]:
        return (self.loss_labels, self.loss_boxes, self.loss_masks)  # type: ignore[return-value]

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        mask_dict: dict,
    ) -> dict[str, Tensor]:
        """Compute the losses.

        Args:
            outputs (dict[str, Tensor]): dict of model outputs.
            targets (list[dict[str, Tensor]]): list of targets.
            mask_dict (dict): dict containing denoise annotation information.

        Returns:
            dict[str, Tensor]: dict containing the computed losses.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        if mask_dict is not None:
            output_known_lbs_bboxes, _, single_pad, scalar = self.prep_for_dn(mask_dict)
            exc_idx = []
            for target in targets:
                device = target["labels"].device
                if len(target["labels"]) > 0:
                    t = torch.arange(0, len(target["labels"]), device=device).long()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar), device=device) * single_pad).long().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([], device=device).long()
                exc_idx.append((output_idx, tgt_idx))

        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        device = next(iter(outputs.values())).device
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=device,
        )
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        num_masks = torch.clamp(num_masks / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self._available_losses:
            losses.update(loss(outputs, targets, indices, num_masks))

        if mask_dict is not None:
            l_dict = {}
            for loss in self._available_losses:
                l_dict.update(loss(output_known_lbs_bboxes, targets, exc_idx, num_masks * scalar))
            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = {
                "loss_bbox_dn": torch.as_tensor(0.0, device=device),
                "loss_giou_dn": torch.as_tensor(0.0, device=device),
                "loss_ce_dn": torch.as_tensor(0.0, device=device),
                "loss_mask_dn": torch.as_tensor(0.0, device=device),
                "loss_dice_dn": torch.as_tensor(0.0, device=device),
            }
            losses.update(l_dict)

        # In case of auxiliary losses, this repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self._available_losses:
                    l_dict = loss(aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                start = 0 if "interm_outputs" in outputs else 1
                if i >= start:
                    if mask_dict is not None:
                        out_ = output_known_lbs_bboxes["aux_outputs"][i]
                        l_dict = {}
                        for loss in self._available_losses:
                            l_dict.update(loss(out_, targets, exc_idx, num_masks * scalar))
                        l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    else:
                        l_dict = {
                            "loss_bbox_dn": torch.as_tensor(0.0, device=device),
                            "loss_giou_dn": torch.as_tensor(0.0, device=device),
                            "loss_ce_dn": torch.as_tensor(0.0, device=device),
                            "loss_mask_dn": torch.as_tensor(0.0, device=device),
                            "loss_dice_dn": torch.as_tensor(0.0, device=device),
                        }
                        losses.update(l_dict)

        # intermediate losses
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            indices = self.matcher(interm_outputs, targets)
            for loss in self._available_losses:
                l_dict = loss(interm_outputs, targets, indices, num_masks)
                l_dict = {k + "_interm": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
