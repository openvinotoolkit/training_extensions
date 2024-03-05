# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for rotated detection lightning module used in OTX."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import torch
from datumaro import Polygon
from torchvision import tv_tensors

from otx.algo.instance_segmentation.otx_instseg_evaluation import (
    OTXMaskRLEMeanAveragePrecision,
)
from otx.core.data.entity.instance_segmentation import (
    InstanceSegBatchPredEntity,
)
from otx.core.model.entity.rotated_detection import OTXRotatedDetModel
from otx.core.model.module.instance_segmentation import OTXInstanceSegLitModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class OTXRotatedDetLitModule(OTXInstanceSegLitModule):
    """Base class for the lightning module used in OTX rotated detection task."""

    def __init__(
        self,
        otx_model: OTXRotatedDetModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable = lambda: OTXMaskRLEMeanAveragePrecision(),
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
        )

    def predict_step(self, *args: torch.Any, **kwargs: torch.Any) -> InstanceSegBatchPredEntity:
        """Predict step for rotated detection task.

        Note: This method is overridden to convert masks to rotated bounding boxes.

        Returns:
            InstanceSegBatchPredEntity: The predicted polygons (rboxes), scores, labels, masks.
        """
        preds = super().predict_step(*args, **kwargs)

        batch_scores: list[torch.Tensor] = []
        batch_bboxes: list[tv_tensors.BoundingBoxes] = []
        batch_labels: list[torch.LongTensor] = []
        batch_polygons: list[list[Polygon]] = []
        batch_masks: list[tv_tensors.Mask] = []

        for img_info, pred_bboxes, pred_scores, pred_labels, pred_masks in zip(
            preds.imgs_info,
            preds.bboxes,
            preds.scores,
            preds.labels,
            preds.masks,
        ):
            boxes = []
            scores = []
            labels = []
            masks = []
            polygons = []

            for bbox, score, label, mask in zip(pred_bboxes, pred_scores, pred_labels, pred_masks):
                if mask.sum() == 0:
                    continue
                np_mask = mask.detach().cpu().numpy().astype(int)
                contours, hierarchies = cv2.findContours(np_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchies is None:
                    continue
                rbox_polygons = []
                for contour, hierarchy in zip(contours, hierarchies[0]):
                    # skip inner contours
                    if hierarchy[3] != -1 or len(contour) <= 2:
                        continue
                    rbox_points = Polygon(cv2.boxPoints(cv2.minAreaRect(contour)).reshape(-1))
                    rbox_polygons.append((rbox_points, rbox_points.get_area()))

                # select the largest polygon
                if len(rbox_polygons) > 0:
                    rbox_polygons.sort(key=lambda x: x[1], reverse=True)
                    polygons.append(rbox_polygons[0][0])
                    scores.append(score)
                    boxes.append(bbox)
                    labels.append(label)
                    masks.append(mask)

            if len(boxes):
                scores = torch.stack(scores)
                boxes = tv_tensors.BoundingBoxes(torch.stack(boxes), format="XYXY", canvas_size=img_info.ori_shape)
                labels = torch.stack(labels)
                masks = torch.stack(masks)

            batch_scores.append(scores)
            batch_bboxes.append(boxes)
            batch_labels.append(labels)
            batch_polygons.append(polygons)
            batch_masks.append(masks)

        return InstanceSegBatchPredEntity(
            batch_size=preds.batch_size,
            images=preds.images,
            imgs_info=preds.imgs_info,
            scores=batch_scores,
            bboxes=batch_bboxes,
            masks=batch_masks,
            polygons=batch_polygons,
            labels=batch_labels,
        )
