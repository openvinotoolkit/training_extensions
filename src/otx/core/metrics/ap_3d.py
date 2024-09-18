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
from otx.core.data.dataset.kitti_eval_python.eval import get_coco_eval_result, get_official_eval_result

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
        # gt_annos = self.targets[0].prepare_kitti_format()[:len(self.preds)]
        current_classes = self.label_info.label_names
        mAP3d, mAP3d_R40 = get_coco_eval_result(
        # mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40  = get_official_eval_result(
            gt_annos,
            self.preds,
            current_classes=0,
        )
        return {"mAP_3d": Tensor([mAP3d]), "mAP_3d_R40": Tensor([mAP3d_R40])}
        # return {"mAP_3d": Tensor([mAP3d]), "mAP_bbox_2d": Tensor([mAPbbox]), "mAP_bev": Tensor([mAPbev]), "mAP_aos": Tensor([mAPaos]),
                # "mAP_bbox_R40": Tensor([mAPbbox_R40]), "mAP_bev_R40": Tensor([mAPbev_R40]), "mAP_3d_R40": Tensor([mAP3d_R40]), "mAP_aos_R40": Tensor([mAPaos_R40])}

    @staticmethod
    def _prepare_annos_for_kitti_metric(targets):
        gt_annos = []
        for images in targets:
            annos = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score' : [],
            }
            for obj in images:
                annos["name"].append(obj.cls_type)
                annos["truncated"].append(obj.trucation)
                annos["occluded"].append(obj.occlusion)
                annos["alpha"].append(obj.alpha)
                annos["bbox"].append(obj.box2d)
                annos["dimensions"].append([obj.h, obj.w, obj.l ])
                annos["location"].append(obj.pos)
                annos["rotation_y"].append(obj.ry)
                annos["score"].append(obj.score)

            for key in annos:
                annos[key] = np.array(annos[key])

            gt_annos.append(annos)

        return gt_annos

def _kitti_metric_measure_callable(label_info: LabelInfo) -> KittiMetric:
    return KittiMetric(label_info=label_info)


KittiMetricCallable = _kitti_metric_measure_callable
