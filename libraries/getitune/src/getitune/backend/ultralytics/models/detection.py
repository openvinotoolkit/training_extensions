# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics detection model."""

from __future__ import annotations

from typing import ClassVar

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.trainers.detection import DetectionTrainer
from getitune.backend.ultralytics.validators.detection import DetectionValidator

from .base import UltralyticsModel


class UltralyticsDetectionModel(UltralyticsModel):
    """YOLO detection model.

    Supported variants:

    * YOLO26: ``yolo26n``, ``yolo26s``, ``yolo26m``, ``yolo26l``, ``yolo26x``
    * YOLO11: ``yolo11n``, ``yolo11s``, ``yolo11m``, ``yolo11l``, ``yolo11x``
    * YOLO12: ``yolo12n``, ``yolo12s``, ``yolo12m``, ``yolo12l``, ``yolo12x``
    """

    task: ClassVar[str] = "detect"
    trainer_cls: ClassVar[type] = DetectionTrainer
    validator_cls: ClassVar[type] = DetectionValidator

    _BASE_URL: ClassVar[str] = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

    _pretrained_weights: ClassVar[dict[str, str]] = {
        # YOLO26
        "yolo26n": f"{_BASE_URL}/yolo26n.pt",
        "yolo26s": f"{_BASE_URL}/yolo26s.pt",
        "yolo26m": f"{_BASE_URL}/yolo26m.pt",
        "yolo26l": f"{_BASE_URL}/yolo26l.pt",
        "yolo26x": f"{_BASE_URL}/yolo26x.pt",
        # YOLO11
        "yolo11n": f"{_BASE_URL}/yolo11n.pt",
        "yolo11s": f"{_BASE_URL}/yolo11s.pt",
        "yolo11m": f"{_BASE_URL}/yolo11m.pt",
        "yolo11l": f"{_BASE_URL}/yolo11l.pt",
        "yolo11x": f"{_BASE_URL}/yolo11x.pt",
        # YOLO12
        "yolo12n": f"{_BASE_URL}/yolo12n.pt",
        "yolo12s": f"{_BASE_URL}/yolo12s.pt",
        "yolo12m": f"{_BASE_URL}/yolo12m.pt",
        "yolo12l": f"{_BASE_URL}/yolo12l.pt",
        "yolo12x": f"{_BASE_URL}/yolo12x.pt",
    }

    metric_keys: ClassVar[dict[str, str]] = {
        "metrics/mAP50(B)": "val/map_50",
        "metrics/mAP50-95(B)": "val/map",
        "metrics/precision(B)": "val/precision",
        "metrics/recall(B)": "val/recall",
        "train/box_loss": "train/loss_bbox",
        "train/cls_loss": "train/loss_cls",
        "train/dfl_loss": "train/loss_dfl",
        "lr/pg0": "lr",
    }

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Per-variant preprocessing defaults.

        All supported detection variants use 640x640 input. The mean/std
        are identity since no additional normalization is needed after
        intensity scaling.
        """
        default = DataInputParams(
            input_size=(640, 640),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
        return dict.fromkeys(self._pretrained_weights, default)
