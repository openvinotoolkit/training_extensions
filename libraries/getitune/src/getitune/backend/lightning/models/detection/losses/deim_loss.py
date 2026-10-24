# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DEIM criterion implementations. Modified from https://github.com/ShihuaHuang95/DEIM."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as f
from torch import Tensor
from torchvision.ops import box_convert

from getitune.backend.lightning.models.common.utils.bbox_overlaps import bbox_overlaps

from .dfine_loss import DFINECriterion


class DEIMCriterion(DFINECriterion):
    """DEIM criterion for DEIM-DFine model."""

    def loss_labels_mal(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[int, int]],
        num_boxes: int,
    ) -> dict[str, Tensor]:
        """Matchability-Aware Loss (MAL) for label prediction.

        Args:
            outputs (dict[str, Tensor]): Model outputs.
            targets (List[Dict[str, Tensor]]): List of target dictionaries.
            indices (List[Tuple[int, int]]): List of tuples of indices.
            num_boxes (int): Number of predicted boxes.

        Returns:
            dict[str, Tensor]: The loss dictionary.
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
        target = f.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = f.sigmoid(src_logits).detach()

        # MAL loss
        target_score = target_score.pow(self.gamma)
        weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss = f.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    @property
    def _available_losses(self) -> tuple[Callable]:
        return (self.loss_boxes, self.loss_labels_vfl, self.loss_labels_mal, self.loss_local)  # type: ignore[return-value]
