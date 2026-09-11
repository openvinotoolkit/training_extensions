# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom semantic segmentation trainer bridging getitune DataModule to Ultralytics."""

from __future__ import annotations

import multiprocessing
from copy import copy
from typing import TYPE_CHECKING, Any, ClassVar

from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.semantic import SemanticSegmentationTrainer as _UltralyticsSemanticSegmentationTrainer

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import semantic_collate_fn
from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin
from getitune.backend.ultralytics.validators.semantic_segmentation import SemanticSegmentationValidator

from .base import GetiTuneBaseTrainer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

_MP_CONTEXT = multiprocessing.get_context("spawn")


class SemanticSegmentationTrainer(GetiTuneBaseTrainer, XPUAwareTrainerMixin, _UltralyticsSemanticSegmentationTrainer):
    """Semantic segmentation trainer that routes data through a getitune DataModule.

    When ``_datamodule`` is set (via the engine's dynamic subclass), data
    loading uses the getitune pipeline and ``preprocess_batch`` skips the
    upstream device transfer (handled by :meth:`_move_batch_to_device`).
    Falls back to default Ultralytics loading otherwise.

    Inherits :class:`XPUAwareTrainerMixin` for Intel XPU device support.
    """

    _task_kind: ClassVar[str] = "semantic"

    def get_dataset(self) -> dict[str, Any]:
        """Return the data config dict from the DataModule bridge.

        The upstream semantic segmentation trainer appends a synthetic
        background class for polygon-based datasets.  getitune's semantic
        segmentation DataModule already provides dense masks with explicit
        background pixels, so we skip that step to keep ``nc`` equal to the
        number of labelled classes.
        """
        if not self._use_getitune_data:
            return super().get_dataset()  # type: ignore[misc]
        return GetiTuneBaseTrainer.get_dataset(self)

    def build_dataset(  # pyrefly: ignore[bad-override]
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
    ) -> UltralyticsDatasetAdapter:
        """Return a semantic segmentation adapter wrapping the appropriate DataModule subset.

        Args:
            img_path: Ignored when DataModule is set.
            mode: ``"train"`` or ``"val"``.
            batch: Batch size (unused by adapter).
        """
        if not self._use_getitune_data:
            return super().build_dataset(img_path, mode, batch)  # type: ignore[misc]

        if self._datamodule is None:
            msg = "DataModule is required when _use_getitune_data=True"
            raise RuntimeError(msg)

        subset_key = (
            self._datamodule.train_subset.subset_name if mode == "train" else self._datamodule.val_subset.subset_name
        )
        vision_dataset = self._datamodule.subsets[subset_key]  # type: ignore[union-attr]
        return UltralyticsDatasetAdapter(vision_dataset, task_kind="semantic")

    def get_dataloader(  # pyrefly: ignore[bad-override]
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> DataLoader:
        """Build a DataLoader from the semantic segmentation adapter dataset."""
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
            collate_fn=semantic_collate_fn,
            pin_memory=True,
            drop_last=False,
            multiprocessing_context=_MP_CONTEXT if nw > 0 else None,
            persistent_workers=nw > 0,
            worker_init_fn=seed_worker,
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Use upstream preprocessing unless the DataModule bridge is active."""
        if not self._use_getitune_data:
            return _UltralyticsSemanticSegmentationTrainer.preprocess_batch(self, batch)
        return self._move_batch_to_device(batch)

    def get_validator(self) -> SemanticSegmentationValidator:
        """Return a custom validator that handles pre-normalised images."""
        if not self._use_getitune_data:
            return super().get_validator()  # type: ignore[return-value]

        self.loss_names = ["ce_loss", "dice_loss", "aux_loss"]  # pyrefly: ignore[bad-assignment]
        validator = SemanticSegmentationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.datamodule = self._datamodule
        return validator
