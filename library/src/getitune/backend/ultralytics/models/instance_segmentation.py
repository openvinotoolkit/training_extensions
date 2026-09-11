# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics instance-segmentation model."""

from __future__ import annotations

from typing import ClassVar

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.trainers.instance_segmentation import SegmentationTrainer
from getitune.backend.ultralytics.validators.instance_segmentation import SegmentationValidator
from getitune.types.export import TaskLevelExportParameters

from .base import UltralyticsModel


class UltralyticsInstSegModel(UltralyticsModel):
    """YOLO instance-segmentation model.

    Supported variants:

    * YOLO26: ``yolo26n-seg``, ``yolo26s-seg``, ``yolo26m-seg``, ``yolo26l-seg``, ``yolo26x-seg``
    * YOLO11: ``yolo11n-seg``, ``yolo11s-seg``, ``yolo11m-seg``, ``yolo11l-seg``, ``yolo11x-seg``
    """

    task: ClassVar[str] = "segment"
    trainer_cls: ClassVar[type] = SegmentationTrainer
    validator_cls: ClassVar[type] = SegmentationValidator

    _BASE_URL: ClassVar[str] = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

    _pretrained_weights: ClassVar[dict[str, str]] = {
        # YOLO26
        "yolo26n-seg": f"{_BASE_URL}/yolo26n-seg.pt",
        "yolo26s-seg": f"{_BASE_URL}/yolo26s-seg.pt",
        "yolo26m-seg": f"{_BASE_URL}/yolo26m-seg.pt",
        "yolo26l-seg": f"{_BASE_URL}/yolo26l-seg.pt",
        "yolo26x-seg": f"{_BASE_URL}/yolo26x-seg.pt",
        # YOLO11
        "yolo11n-seg": f"{_BASE_URL}/yolo11n-seg.pt",
        "yolo11s-seg": f"{_BASE_URL}/yolo11s-seg.pt",
        "yolo11m-seg": f"{_BASE_URL}/yolo11m-seg.pt",
        "yolo11l-seg": f"{_BASE_URL}/yolo11l-seg.pt",
        "yolo11x-seg": f"{_BASE_URL}/yolo11x-seg.pt",
    }

    metric_keys: ClassVar[dict[str, str]] = {
        "metrics/mAP50(M)": "val/map_50",
        "metrics/mAP50-95(M)": "val/map",
        "metrics/precision(B)": "val/precision",
        "metrics/recall(B)": "val/recall",
        "train/box_loss": "train/loss_bbox",
        "train/cls_loss": "train/loss_cls",
        "train/dfl_loss": "train/loss_dfl",
        "train/seg_loss": "train/loss_mask",
    }

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Per-variant preprocessing defaults.

        Mean/std are identity (no additional normalization after
        intensity scaling).
        """
        default = DataInputParams(
            input_size=(640, 640),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
        return dict.fromkeys(self._pretrained_weights, default)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Instance segmentation export parameters.

        ``nms_execute`` is set when the model is exported without built-in NMS
        so that ModelAPI performs NMS during post-processing.
        """
        conf = self._export_args.get("confidence_threshold")
        if conf is None:
            conf = self.extra_overrides.get("conf", 0.25)
        iou = self._export_args.get("iou_threshold")
        if iou is None:
            iou = self.extra_overrides.get("iou", 0.5)
        return TaskLevelExportParameters(
            model_type="YOLO-seg",
            model_name=self.model_name,
            task_type="instance_segmentation",
            label_info=self.label_info,
            optimization_config={},
            confidence_threshold=float(conf),
            iou_threshold=float(iou),
            nms_execute=not self.export_nms,
        )
