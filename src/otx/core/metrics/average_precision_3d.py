# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX  metric used for 3D object detection tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor
import torch
from torchmetrics import Metric

from otx.core.metrics.kitti_3d_eval import get_coco_eval_result
from torchmetrics.detection.mean_ap import MeanAveragePrecision

if TYPE_CHECKING:
    import numpy as np

    from otx.core.types.label import LabelInfo


class KittiMetric(Metric):
    """Computes the 2D/3D average precision (coco style) for object detection 3d task.

    Args:
        label_info (int): Dataclass including label information.
    """

    def __init__(
        self,
        label_info: LabelInfo,
    ):
        super().__init__()

        self.label_info: LabelInfo = label_info
        self.mean_ap: MeanAveragePrecision = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        self.reset()

    def reset(self) -> None:
        """Reset for every validation and test epoch.

        Please be careful that some variables should not be reset for each epoch.
        """
        super().reset()
        self.preds: list[dict[str, np.array]] = []
        self.targets: list[dict[str, np.array]] = []

    def update(self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]) -> None:
        """Update total predictions and targets from given batch predicitons and targets."""
        self.preds.extend(preds)
        self.targets.extend(target)

    def compute(self) -> dict:
        """Compute metrics for 3d object detection."""
        current_classes = self.label_info.label_names
        preds_for_torchmetrics = self.prepare_targets(self.preds)
        targets_for_torchmetrics = self.prepare_targets(self.targets)
        map_bbox_coco = self.mean_ap(preds_for_torchmetrics, targets_for_torchmetrics)
        map_3d = get_coco_eval_result(
            self.targets,
            self.preds,
            current_classes=[curcls.lower() for curcls in current_classes],
        )
        breakpoint()
        # use moderate difficulty as final score. Average across all calsses.
        return {"mAP_bbox_3d": Tensor([map_3d[:, 1].mean()]), "mAP_bbox_2d": map_bbox_coco["map_50"]}

    def prepare_inputs_for_map_coco(self, targets: list[dict[str, np.array]]) -> list[dict[str, Tensor]]:
        """Prepare targets for torchmetrics."""
        targets_for_torchmetrics = []
        for target in targets:
            targets_for_torchmetrics.append(
                {
                    "boxes": torch.tensor(target["bbox"]),
                    "scores": torch.tensor(target["score"]) if "score" in target else None,
                    "labels": torch.tensor([self.label_info.label_names.index(label) for label in target["name"]], dtype=torch.long),
                }
            )
        return targets_for_torchmetrics

def _kitti_metric_measure_callable(label_info: LabelInfo) -> KittiMetric:
    return KittiMetric(label_info=label_info)


KittiMetricCallable = _kitti_metric_measure_callable
