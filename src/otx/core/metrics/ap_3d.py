# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX  metric used for 3D object detection tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from torch import Tensor
from torchmetrics import Metric

from otx.core.metrics.kitti_3d_eval import get_coco_eval_result

if TYPE_CHECKING:
    from otx.core.types.label import LabelInfo


class KittiMetric(Metric):
    """Computes the f-measure (also known as F1-score) for a resultset.

    The f-measure is typically used in detection (localization) tasks to obtain a single number that balances precision
    and recall.

    To determine whether a predicted box matches a ground truth box an overlap measured
    is used based on a minimum
    intersection-over-union (IoU), by default a value of 0.5 is used.

    In addition spurious results are eliminated by applying non-max suppression (NMS) so that two predicted boxes with
    IoU > threshold are reduced to one. This threshold can be determined automatically by setting `vary_nms_threshold`
    to True.

    Args:
        label_info (int): Dataclass including label information.
        vary_nms_threshold (bool): if True the maximal F-measure is determined by optimizing for different NMS threshold
            values. Defaults to False.
        cross_class_nms (bool): Whether non-max suppression should be applied cross-class. If True this will eliminate
            boxes with sufficient overlap even if they are from different classes. Defaults to False.
    """

    def __init__(
        self,
        label_info: LabelInfo,
    ):
        super().__init__()

        self.label_info: LabelInfo = label_info
        self.reset()

    def reset(self) -> None:
        """Reset for every validation and test epoch.

        Please be careful that some variables should not be reset for each epoch.
        """
        super().reset()
        self.preds: list[dict[str, np.array]] = []
        self.targets: list[np.ndarray] = []

    def update(self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]) -> None:
        """Update total predictions and targets from given batch predicitons and targets."""
        self.preds.extend(preds)
        self.targets.extend(target)

    def compute(self) -> dict:
        """Compute metrics for 3d object detection."""
        gt_annos = self._prepare_annos_for_kitti_metric(self.targets)
        current_classes = self.label_info.label_names

        map_bbox, map_3d = get_coco_eval_result(
            gt_annos,
            self.preds,
            current_classes=[curcls.lower() for curcls in current_classes],
        )
        # use moderate difficulty as final score. Average across all calsses.
        return {"mAP_bbox_3d": Tensor([map_3d[:, 1].mean()]), "mAP_bbox_2d": Tensor([map_bbox[:, 1].mean()])}

    @staticmethod
    def _prepare_annos_for_kitti_metric(targets: list[Any]) -> list[dict[str, np.array]]:
        gt_annos = []
        for images in targets:
            names = []
            alphas = []
            bboxes = []
            dimensions = []
            locations = []
            rotation_y = []
            occlusions = []
            truncations = []
            scores = []

            for obj in images:
                names.append(obj.cls_type)
                alphas.append(obj.alpha)
                bboxes.append(obj.box2d)
                dimensions.append([obj.l, obj.h, obj.w])
                locations.append(obj.pos)
                rotation_y.append(obj.ry)
                truncations.append(obj.trucation)
                occlusions.append(obj.occlusion)
                scores.append(obj.score)

            annos = {
                "name": np.array(names),
                "alpha": np.array(alphas),
                "bbox": np.array(bboxes).reshape(-1, 4),
                "dimensions": np.array(dimensions).reshape(-1, 3),
                "location": np.array(locations).reshape(-1, 3),
                "rotation_y": np.array(rotation_y),
                "occluded": np.array(occlusions),
                "truncated": np.array(truncations),
                "score": np.array(scores),
            }

            gt_annos.append(annos)

        return gt_annos


def _kitti_metric_measure_callable(label_info: LabelInfo) -> KittiMetric:
    return KittiMetric(label_info=label_info)


KittiMetricCallable = _kitti_metric_measure_callable
