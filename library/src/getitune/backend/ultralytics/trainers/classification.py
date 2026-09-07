# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom classification trainer bridging getitune DataModule to Ultralytics."""

from __future__ import annotations

import multiprocessing
from copy import copy
from typing import TYPE_CHECKING, Any, ClassVar

from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.classify import ClassificationTrainer as _UltralyticsClassificationTrainer
from ultralytics.utils.torch_utils import unwrap_model

from getitune.backend.ultralytics.data.collate import classification_collate_fn, multilabel_collate_fn
from getitune.backend.ultralytics.models.criterion import MultiLabelClassificationLoss
from getitune.backend.ultralytics.models.utils import patch_multilabel_classify_head
from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin
from getitune.backend.ultralytics.validators.classification import (
    ClassificationValidator,
    MultiLabelClassificationValidator,
)

from .base import GetiTuneBaseTrainer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

_MP_CONTEXT = multiprocessing.get_context("spawn")


class ClassificationTrainer(  # pyrefly: ignore[inconsistent-inheritance]
    GetiTuneBaseTrainer, XPUAwareTrainerMixin, _UltralyticsClassificationTrainer
):
    """Classification trainer that routes data through a getitune DataModule.

    When ``_datamodule`` is set (via the engine's dynamic subclass), data
    loading uses the getitune pipeline and ``preprocess_batch`` skips the
    upstream device transfer (handled by :meth:`_move_batch_to_device`).
    Falls back to default Ultralytics loading otherwise.

    Inherits :class:`XPUAwareTrainerMixin` for Intel XPU device support.
    """

    _task_kind: ClassVar[str] = "classify"

    def get_dataloader(  # pyrefly: ignore[bad-override]
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> DataLoader:
        """Build a DataLoader from the classification adapter dataset."""
        if not self._use_getitune_data:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)  # type: ignore[misc]

        dataset = self.build_dataset(dataset_path, mode, batch_size)
        nw: int = self.args.workers  # type: ignore[attr-defined]
        shuffle = mode == "train"
        return InfiniteDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            prefetch_factor=4 if nw > 0 else None,
            collate_fn=classification_collate_fn,
            pin_memory=True,
            drop_last=False,
            multiprocessing_context=_MP_CONTEXT if nw > 0 else None,
            persistent_workers=nw > 0,
            worker_init_fn=seed_worker,
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Use upstream preprocessing unless the DataModule bridge is active."""
        if not self._use_getitune_data:
            return _UltralyticsClassificationTrainer.preprocess_batch(self, batch)
        return self._move_batch_to_device(batch)

    def get_validator(self) -> ClassificationValidator:
        """Return a custom validator that handles pre-normalised images."""
        if not self._use_getitune_data:
            return super().get_validator()  # type: ignore[return-value]

        self.loss_names = ["loss"]  # pyrefly: ignore[bad-assignment]
        validator = ClassificationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.datamodule = self._datamodule
        return validator


class MultiLabelClassificationTrainer(  # pyrefly: ignore[inconsistent-inheritance]
    GetiTuneBaseTrainer, XPUAwareTrainerMixin, _UltralyticsClassificationTrainer
):
    """Multi-label classification trainer with BCE loss and sigmoid inference."""

    _task_kind: ClassVar[str] = "multilabel"

    def get_dataloader(  # pyrefly: ignore[bad-override]
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> DataLoader:
        """Build a DataLoader from the multi-label adapter dataset."""
        if not self._use_getitune_data:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)  # type: ignore[misc]

        dataset = self.build_dataset(dataset_path, mode, batch_size)
        nw: int = self.args.workers  # type: ignore[attr-defined]
        shuffle = mode == "train"
        return InfiniteDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            prefetch_factor=4 if nw > 0 else None,
            collate_fn=multilabel_collate_fn,
            pin_memory=True,
            drop_last=False,
            multiprocessing_context=_MP_CONTEXT if nw > 0 else None,
            persistent_workers=nw > 0,
            worker_init_fn=seed_worker,
        )

    def setup_model(self) -> None:
        """Build the classification model and rewire it for multi-label."""
        super().setup_model()
        patch_multilabel_classify_head(unwrap_model(self.model))
        unwrap_model(self.model).criterion = MultiLabelClassificationLoss()

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Use upstream preprocessing unless the DataModule bridge is active."""
        if not self._use_getitune_data:
            return _UltralyticsClassificationTrainer.preprocess_batch(self, batch)
        return self._move_batch_to_device(batch)

    def get_validator(self) -> MultiLabelClassificationValidator:
        """Return a custom validator that computes multi-label metrics."""
        if not self._use_getitune_data:
            return super().get_validator()  # type: ignore[return-value]

        self.loss_names = ["loss"]  # pyrefly: ignore[bad-assignment]
        validator = MultiLabelClassificationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.datamodule = self._datamodule
        return validator
