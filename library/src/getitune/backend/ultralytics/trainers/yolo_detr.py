# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom YOLO-DETR trainer bridging getitune data to Ultralytics."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any, cast

from ultralytics.models.rtdetr.train import DEIMTrainer as _DEIMTrainer
from ultralytics.models.rtdetr.train import RTDETRTrainer as _RTDETRTrainer
from ultralytics.utils.torch_utils import unwrap_model

from getitune.backend.ultralytics.data.collate import detection_collate_fn
from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin
from getitune.backend.ultralytics.validators.yolo_detr import YoloDetrValidator

from .base import GetiTuneBaseTrainer

if TYPE_CHECKING:
    from torch import nn
    from ultralytics.models.rtdetr.val import RTDETRValidator as _RTDETRValidator


class YoloDetrTrainer(
    GetiTuneBaseTrainer,
    XPUAwareTrainerMixin,
    _DEIMTrainer,
):
    """YOLO-DETR trainer using getitune's DataModule bridge and XPU support."""

    _collate_fn = staticmethod(detection_collate_fn)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preserve getitune-normalized images while retaining upstream preprocessing otherwise."""
        if not self._use_getitune_data:
            return _DEIMTrainer.preprocess_batch(self, batch)
        return self._move_batch_to_device(batch)

    def train(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Skip upstream epoch callbacks that require its native dataset type."""
        if not self._use_getitune_data:
            return super().train(*args, **kwargs)

        if self.args.close_mosaic:
            self.args.close_mosaic = 0
        return _RTDETRTrainer.train(self, *args, **kwargs)

    def get_validator(self) -> _RTDETRValidator:
        """Return the getitune-aware YOLO-DETR validator."""
        if not self._use_getitune_data:
            return super().get_validator()

        model_layers = cast("nn.Sequential", unwrap_model(self.model).model)
        head_name = type(model_layers[-1]).__name__
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        if head_name == "DeimDecoder":
            loss_names += ["fgl_loss", "ddf_loss"]
        self.loss_names = tuple(loss_names)  # pyrefly: ignore[bad-assignment]

        validator = YoloDetrValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.datamodule = self._datamodule
        return validator
