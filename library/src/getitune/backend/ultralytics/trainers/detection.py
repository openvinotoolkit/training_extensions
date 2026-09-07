# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom detection trainer bridging getitune DataModule to Ultralytics."""

from __future__ import annotations

from copy import copy
from typing import Any, ClassVar

from ultralytics.models.yolo.detect import DetectionTrainer as _UltralyticsDetectionTrainer
from ultralytics.models.yolo.detect import DetectionValidator as _UltralyticsDetectionValidator

from getitune.backend.ultralytics.data.collate import detection_collate_fn
from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin
from getitune.backend.ultralytics.validators.detection import DetectionValidator

from .base import GetiTuneBaseTrainer


class DetectionTrainer(GetiTuneBaseTrainer, XPUAwareTrainerMixin, _UltralyticsDetectionTrainer):
    """Detection trainer that routes data through a getitune DataModule.

    When ``_datamodule`` is set (via the engine's dynamic subclass),
    data loading uses the getitune pipeline and ``preprocess_batch``
    skips normalisation.  Falls back to default Ultralytics loading otherwise.

    Inherits :class:`XPUAwareTrainerMixin` for Intel XPU device support.
    """

    _task_kind: ClassVar[str] = "detect"
    _collate_fn = staticmethod(detection_collate_fn)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Use upstream preprocessing unless the DataModule bridge is active."""
        if not self._use_getitune_data:
            return _UltralyticsDetectionTrainer.preprocess_batch(self, batch)
        return self._move_batch_to_device(batch)

    def get_validator(self) -> _UltralyticsDetectionValidator:
        """Return a custom validator that handles pre-normalised images."""
        if not self._use_getitune_data:
            return super().get_validator()

        self.loss_names = ["box_loss", "cls_loss", "dfl_loss"]  # pyrefly: ignore[bad-assignment]
        validator = DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.datamodule = self._datamodule
        return validator
