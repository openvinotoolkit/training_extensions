# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Customised MAP metric for instance segmentation."""

from __future__ import annotations

from typing import Any

import pycocotools.mask as mask_utils
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class OTXInstSegMeanAveragePrecision(MeanAveragePrecision):
    """Mean Average Precision for Instance Segmentation.

    This metric computes RLE directly to accelerate the computation.
    """

    def update(self, preds: list[dict], target: list[dict]) -> None:
        """Update the metric with the given predictions and targets.

        Args:
            preds (list[dict]): list of RLE encoded masks
            target (list[dict]): list of RLE encoded masks
        """
        for item in preds:
            bbox_detection, mask_detection = self._get_safe_item_values(item, warn=self.warn_on_many_detections)
            if bbox_detection is not None:
                self.detection_box.append(bbox_detection)
            if mask_detection is not None:
                self.detection_mask.append(mask_detection)
            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            bbox_groundtruth, mask_groundtruth = self._get_safe_item_values(item)
            if bbox_groundtruth is not None:
                self.groundtruth_box.append(bbox_groundtruth)
            if mask_groundtruth is not None:
                self.groundtruth_mask.append(mask_groundtruth)
            self.groundtruth_labels.append(item["labels"])
            self.groundtruth_crowds.append(item.get("iscrowd", torch.zeros_like(item["labels"])))
            self.groundtruth_area.append(item.get("area", torch.zeros_like(item["labels"])))

    def _get_safe_item_values(
        self,
        item: dict[str, Any],
        warn: bool = False,
    ) -> tuple:
        """Convert masks to RLE format.

        Args:
            item: input dictionary containing the boxes or masks
            warn: whether to warn if the number of boxes or masks exceeds the max_detection_thresholds

        Returns:
            RLE encoded masks
        """
        if "segm" in self.iou_type:
            masks = []
            for rle in item["masks"]:
                if isinstance(rle["counts"], list):
                    rle["counts"] = mask_utils.frPyObjects(rle, *rle["size"])["counts"]
                masks.append((tuple(rle["size"]), rle["counts"]))
        return None, tuple(masks)
