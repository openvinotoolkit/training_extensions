# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX  metric used for classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numba
from torch import Tensor
from torchmetrics import Metric
from otx.core.data.dataset.kitti_eval_python.eval import get_coco_eval_result

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
        self.preds: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []

    def update(self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]) -> None:
        """Update total predictions and targets from given batch predicitons and targets."""
        breakpoint()
        for pred, tget in zip(preds, target):
            self.preds.extend(
                [
                    (pred["keypoints"], pred["scores"]),
                ],
            )
            self.targets.extend(
                [
                    (tget["keypoints"], tget["keypoints_visible"]),
                ],
            )

    def compute(self) -> dict:
        """Compute metrics for 3d object detection."""
        gt_annos = self.targets
        dt_annos = self.preds
        current_classes = self.label_info.label_names

        mAP_bbox, mAP_bev, mAP_3d, mAP_aos = get_coco_eval_result(
            gt_annos,
            dt_annos,
            current_classes,
        )

        return {"mAP_3d": Tensor([mAP_3d]), "mAP_bbox_2d": Tensor([mAP_bbox]), "mAP_bev": Tensor([mAP_bev]), "mAP_aos": Tensor([mAP_aos])}


def _kitti_metric_measure_callable(label_info: LabelInfo) -> KittiMetric:
    return KittiMetric(label_info=label_info)


KittiMetricCallable = _kitti_metric_measure_callable
