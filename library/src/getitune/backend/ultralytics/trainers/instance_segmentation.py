# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom segmentation trainer bridging getitune DataModule to Ultralytics."""

from __future__ import annotations

from copy import copy
from functools import partial
from typing import Any, ClassVar

import torch
import torch.nn.functional as f
from ultralytics.models.yolo.segment import SegmentationTrainer as _UltralyticsSegmentationTrainer
from ultralytics.models.yolo.segment import SegmentationValidator as _UltralyticsSegmentationValidator

from getitune.backend.ultralytics.data.collate import MASK_RATIO, instance_seg_collate_fn
from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin
from getitune.backend.ultralytics.validators.instance_segmentation import SegmentationValidator

from .base import GetiTuneBaseTrainer


class SegmentationTrainer(GetiTuneBaseTrainer, XPUAwareTrainerMixin, _UltralyticsSegmentationTrainer):
    """Instance-segmentation trainer that routes data through a getitune DataModule.

    Mirrors :class:`DetectionTrainer` but dispatches the adapter to
    ``task_kind="segment"``.  Falls back to default Ultralytics loading otherwise.

    Inherits :class:`XPUAwareTrainerMixin` for Intel XPU device support.
    """

    _task_kind: ClassVar[str] = "segment"
    _collate_fn = partial(instance_seg_collate_fn, mask_ratio=MASK_RATIO)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch for training: move to device and downsample masks.

        GT masks from the DataModule adapter are at full input resolution
        (e.g. 640x640).  The segmentation loss expects masks at proto head
        resolution (input_size / 4 = 160x160) — if masks are larger, the loss
        upsamples the proto features to match, computing gradients at a
        resolution the model cannot actually produce.  This prevents proper
        convergence of mask features.

        Masks are in **overlap index map** format ``(B, H, W)`` where pixel
        values 1..N identify instance ownership (0 = background).

        On the polygon path the collate already rasterizes directly at that
        resolution, so this method is a no-op for masks — the resample below
        only fires for the dense-mask fallback (``annotation_format="mask"``),
        or when ``mask_ratio=1`` is used to keep full-resolution GT. Resizing
        an index map is lossy (nearest interpolation can drop thin instances
        entirely), so not doing it is both faster and more accurate.

        During training we downsample masks to proto resolution (nearest
        interpolation to preserve index values).  Validation keeps full-res
        masks — the validator handles resizing independently.
        """
        if not self._use_getitune_data:
            return _UltralyticsSegmentationTrainer.preprocess_batch(self, batch)

        batch = self._move_batch_to_device(batch)

        masks = batch.get("masks")
        if masks is None or masks.numel() == 0:
            return batch

        img_h, img_w = batch["img"].shape[2:]
        target_h, target_w = img_h // MASK_RATIO, img_w // MASK_RATIO

        if masks.shape[1] == target_h and masks.shape[2] == target_w:
            # Already at target resolution; nothing to do.
            return batch

        # Nearest-neighbour preserves the integer instance indices.
        batch["masks"] = (
            f.interpolate(masks.unsqueeze(1).float(), size=(target_h, target_w), mode="nearest")
            .squeeze(1)
            .to(torch.uint8)
        )

        # Downsample semantic masks to the same spatial resolution.
        if "sem_masks" in batch:
            batch["sem_masks"] = f.interpolate(
                batch["sem_masks"].unsqueeze(1).float(),
                size=(target_h, target_w),
                mode="nearest",
            ).squeeze(1)

        return batch

    def set_model_attributes(self) -> None:
        """Set model attributes; enable overlap mask format when using DataModule."""
        super().set_model_attributes()
        if self._use_getitune_data and hasattr(self.args, "overlap_mask"):
            self.args.overlap_mask = True

    def get_validator(self) -> _UltralyticsSegmentationValidator:
        """Return a custom segmentation validator that handles pre-normalised images."""
        if not self._use_getitune_data:
            return super().get_validator()

        self.loss_names = [  # pyrefly: ignore[bad-assignment]
            "box_loss",
            "seg_loss",
            "cls_loss",
            "dfl_loss",
            "sem_loss",
        ]
        validator = SegmentationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.datamodule = self._datamodule
        return validator
