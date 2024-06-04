# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Customised MAP metric for instance segmentation."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pycocotools.mask as mask_utils
import torch
from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from otx.core.types.label import LabelInfo

from .fmeasure import FMeasure

if TYPE_CHECKING:
    from torchmetrics import Metric


class MaskRLEMeanAveragePrecision(MeanAveragePrecision):
    """Customised MAP metric for instance segmentation.

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


class MaskRLEMeanAveragePrecisionFMeasure(MetricCollection):
    """Computes the mean AP with f-measure for a resultset.

    NOTE: IMPORTANT!!! Do not use this metric to evaluate a F1 score on a test set.
        This is because it can pollute test evaluation.
        It will optimize the confidence threshold on the test set by
        doing line search on confidence threshold axis.
        The correct way to obtain the test set F1 score is to use
        the best confidence threshold obtained from the validation set.
        You should use `--metric otx.core.metrics.fmeasure.FMeasureCallable`override
        to correctly obtain F1 score from a test set.
    """

    def __init__(self, box_format: str, iou_type: str, label_info: LabelInfo, **kwargs):
        map_kwargs = self._filter_kwargs(MaskRLEMeanAveragePrecision, kwargs)
        fmeasure_kwargs = self._filter_kwargs(FMeasure, kwargs)

        super().__init__(
            [
                MaskRLEMeanAveragePrecision(box_format, iou_type, **map_kwargs),
                FMeasure(label_info, **fmeasure_kwargs),
            ],
        )

    def _filter_kwargs(self, cls: type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        cls_params = inspect.signature(cls.__init__).parameters
        valid_keys = set(cls_params.keys()) - {"self"}
        return {k: v for k, v in kwargs.items() if k in valid_keys}


def _mean_ap_callable(label_info: LabelInfo) -> Metric:  # noqa: ARG001
    return MeanAveragePrecision(box_format="xyxy", iou_type="bbox")


MeanAPCallable = _mean_ap_callable


def _mask_rle_mean_ap_callable(label_info: LabelInfo) -> Metric:  # noqa: ARG001
    return MaskRLEMeanAveragePrecision(
        box_format="xyxy",
        iou_type="segm",
    )


def _rle_mean_ap_f_measure_callable(label_info: LabelInfo) -> MaskRLEMeanAveragePrecisionFMeasure:
    return MaskRLEMeanAveragePrecisionFMeasure(
        box_format="xyxy",
        iou_type="segm",
        label_info=label_info,
    )


MaskRLEMeanAPCallable = _mask_rle_mean_ap_callable

MaskRLEMeanAPFMeasureCallable = _rle_mean_ap_f_measure_callable
