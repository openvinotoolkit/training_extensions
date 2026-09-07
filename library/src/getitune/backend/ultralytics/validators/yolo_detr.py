# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""YOLO-DETR validator for the getitune data bridge."""

from __future__ import annotations

from typing import ClassVar

from ultralytics.models.yolodetr.train import YOLODETRValidator as _YOLODETRValidator

from getitune.backend.ultralytics.data.collate import detection_collate_fn

from .base import GetiTuneValidatorMixin


class YoloDetrValidator(GetiTuneValidatorMixin, _YOLODETRValidator):
    """YOLO-DETR validator using getitune's DataModule bridge."""

    _task_kind: ClassVar[str] = "detect"
    _collate_fn = staticmethod(detection_collate_fn)
