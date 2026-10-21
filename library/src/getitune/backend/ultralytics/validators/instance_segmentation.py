# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segmentation validator for the getitune data bridge."""

from __future__ import annotations

from typing import ClassVar

from ultralytics.models.yolo.segment import SegmentationValidator as _UltralyticsSegmentationValidator

from getitune.backend.ultralytics.data.collate import instance_seg_collate_fn

from .base import GetiTuneValidatorMixin


class SegmentationValidator(GetiTuneValidatorMixin, _UltralyticsSegmentationValidator):
    """Instance-segmentation validator for the getitune data bridge."""

    _task_kind: ClassVar[str] = "segment"
    _collate_fn = staticmethod(instance_seg_collate_fn)
