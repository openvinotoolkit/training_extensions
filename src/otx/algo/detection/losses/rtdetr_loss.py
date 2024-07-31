# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RT-Detr loss, modified from https://github.com/lyuwenyu/RT-DETR."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torchvision.ops import box_convert

from otx.algo.common.losses import GIoULoss, L1Loss
from otx.algo.common.utils.bbox_overlaps import bbox_overlaps
from otx.algo.detection.utils.matchers import HungarianMatcher


class DetrCriterion(nn.Module):
    """This class computes the loss for DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)

    Args:
        weight_dict (dict[str, int | float]): A dictionary containing the weights for different loss components.
        alpha (float, optional): The alpha parameter for the loss calculation. Defaults to 0.2.
        gamma (float, optional): The gamma parameter for the loss calculation. Defaults to 2.0.
        num_classes (int, optional): The number of classes. Defaults to 80.
    """

    def __init__(
        self,
        weight_dict: dict[str, int | float],
        alpha: float = 0.2,
        gamma: float = 2.0,
        num_classes: int = 80,
    ) -> None:
        """Create the criterion."""
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2})
        loss_bbox_weight = weight_dict["loss_bbox"] if "loss_bbox" in weight_dict else 1.0
        loss_giou_weight = weight_dict["loss_giou"] if "loss_giou" in weight_dict else 1.0
        self.loss_vfl_weight = weight_dict["loss_vfl"] if "loss_vfl" in weight_dict else 1.0
        self.alpha = alpha
        self.gamma = gamma
        self.lossl1 = L1Loss(loss_weight=loss_bbox_weight)
        self.giou = GIoULoss(loss_weight=loss_giou_weight)

    def loss_labels_vfl(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[int, int]],
        num_boxes: int,
    ) -> dict[str, torch.Tensor]:
        """Compute the vfl loss.

        Args:
            outputs (dict[str, torch.Tensor]): Model outputs.
            targets (List[Dict[str, torch.Tensor]]): List of target dictionaries.
            indices (List[Tuple[int, int]]): List of tuples of indices.
            num_boxes (int): Number of predicted boxes.
        """
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious = bbox_overlaps(
            box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
        )
        ious = torch.diag(ious).detach()

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = nn.functional.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = nn.functional.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = nn.functional.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss * self.loss_vfl_weight}

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[int, int]],
        num_boxes: int,
    ) -> dict[str, torch.Tensor]:
        """Compute the losses re)L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.

        Args:
            outputs (dict[str, torch.Tensor]): The outputs of the model.
            targets (list[dict[str, torch.Tensor]]): The targets.
            indices (list[tuple[int, int]]): The indices of the matched boxes.
            num_boxes (int): The number of boxes.

        Returns:
            dict[str, torch.Tensor]: The losses.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

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

    def _get_src_permutation_idx(
        self,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @property
    def _available_losses(self) -> tuple[Callable]:
        return (self.loss_boxes, self.loss_labels_vfl)  # type: ignore[return-value]

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """This performs the loss computation.

        Args:
             outputs (dict[str, torch.Tensor]): dict of tensors, see the output
                specification of the model for the format
             targets (list[dict[str, torch.Tensor]]): list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if "pred_boxes" not in outputs or "pred_logits" not in outputs:
            msg = "The model should return the predicted boxes and logits"
            raise ValueError(msg)

        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            world_size = torch.distributed.get_world_size()
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self._available_losses:
            l_dict = loss(outputs, targets, indices, num_boxes)
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self._available_losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}

                    l_dict = loss(aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if "dn_aux_outputs" in outputs:
            if "dn_meta" not in outputs:
                msg = "dn_meta is not in outputs"
                raise ValueError(msg)
            indices = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]

            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self._available_losses:
                    kwargs = {}
                    l_dict = loss(aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(
        dn_meta: dict[str, list[torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """get_cdn_matched_indices.

        Args:
            dn_meta (dict[str, list[torch.Tensor]]): meta data for cdn
            targets (list[dict[str, torch.Tensor]]): targets
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                if len(dn_positive_idx[i]) != len(gt_idx):
                    msg = f"len(dn_positive_idx[i]) != len(gt_idx), {len(dn_positive_idx[i])} != {len(gt_idx)}"
                    raise ValueError(msg)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    ),
                )

        return dn_match_indices
