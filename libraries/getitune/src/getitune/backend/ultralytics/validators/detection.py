# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Detection validator for the getitune data bridge."""

from __future__ import annotations

from typing import ClassVar

from ultralytics.models.yolo.detect import DetectionValidator as _UltralyticsDetectionValidator

from getitune.backend.ultralytics.data.collate import detection_collate_fn

from .base import GetiTuneValidatorMixin


class DetectionValidator(GetiTuneValidatorMixin, _UltralyticsDetectionValidator):
    """Detection validator for the getitune data bridge."""

    _task_kind: ClassVar[str] = "detect"
    _collate_fn = staticmethod(detection_collate_fn)
