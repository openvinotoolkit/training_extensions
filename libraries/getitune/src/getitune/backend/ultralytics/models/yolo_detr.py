# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics YOLO-DETR detection model wrapper."""

from __future__ import annotations

from typing import ClassVar

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.trainers.yolo_detr import YoloDetrTrainer
from getitune.backend.ultralytics.validators.yolo_detr import YoloDetrValidator
from getitune.types.export import TaskLevelExportParameters

from .base import UltralyticsModel


class UltralyticsYoloDetrModel(UltralyticsModel):
    """YOLO-DETR model wrapper for checkpoint-backed early-access models."""

    task: ClassVar[str] = "detect"
    trainer_cls: ClassVar[type] = YoloDetrTrainer
    validator_cls: ClassVar[type] = YoloDetrValidator
    _pretrained_weights: ClassVar[dict[str, str]] = {}

    metric_keys: ClassVar[dict[str, str]] = {
        "metrics/mAP50(B)": "val/map_50",
        "metrics/mAP50-95(B)": "val/map",
        "metrics/precision(B)": "val/precision",
        "metrics/recall(B)": "val/recall",
        "train/giou_loss": "train/loss_giou",
        "train/cls_loss": "train/loss_cls",
        "train/l1_loss": "train/loss_l1",
        "train/fgl_loss": "train/loss_fgl",
        "train/ddf_loss": "train/loss_ddf",
        "lr/pg0": "lr",
    }

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Return identity normalization at the standard detection resolution."""
        params = DataInputParams(
            input_size=(640, 640),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
        return {"yolo27x": params}

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Return export metadata for the YOLO-DETR output contract."""
        confidence = self._export_args.get("confidence_threshold")
        if confidence is None:
            confidence = self.extra_overrides.get("conf", 0.5)
        iou = self._export_args.get("iou_threshold")
        if iou is None:
            iou = self.extra_overrides.get("iou", 0.5)
        return TaskLevelExportParameters(
            model_type="YOLODETR",
            model_name=self.model_name,
            task_type="detection",
            label_info=self.label_info,
            optimization_config={},
            confidence_threshold=float(confidence),
            iou_threshold=float(iou),
            nms_execute=False,
        )
