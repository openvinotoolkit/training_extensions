# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Semantic segmentation validator for the getitune data bridge."""

from __future__ import annotations

import multiprocessing
from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader
from ultralytics.models.yolo.semantic import SemanticSegmentationValidator as _UltralyticsSemanticSegmentationValidator

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import semantic_collate_fn

from .base import GetiTuneValidatorMixin


class SemanticSegmentationValidator(GetiTuneValidatorMixin, _UltralyticsSemanticSegmentationValidator):
    """Semantic segmentation validator for the getitune data bridge."""

    _task_kind: ClassVar[str] = "semantic"

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensors to device and record the semantic target shape.

        The upstream semantic segmentation postprocess expects
        ``_semantic_target_shape`` to be set from the ground-truth mask
        resolution.  We reuse the getitune bridge preprocessing (no /255
        normalization) and add the mask shape bookkeeping.
        """
        batch = GetiTuneValidatorMixin.preprocess(self, batch)
        if "semantic_mask" in batch:
            batch["semantic_mask"] = batch["semantic_mask"].to(self.device, dtype=torch.int32)
            self._semantic_target_shape = tuple(batch["semantic_mask"].shape[-2:])
        return batch

    def _build_adapter_dataloader(self) -> DataLoader:
        """Build a semantic segmentation DataLoader from the DataModule's val/test subset."""
        if self._datamodule is None:
            msg = "_build_adapter_dataloader requires a DataModule"
            raise TypeError(msg)
        test_key = self._datamodule.test_subset.subset_name
        val_key = self._datamodule.val_subset.subset_name
        test_subset = self._datamodule.subsets.get(test_key)
        subset = test_subset or self._datamodule.subsets[val_key]
        num_workers = (
            self._datamodule.test_subset.num_workers if test_subset else self._datamodule.val_subset.num_workers
        )
        adapter = UltralyticsDatasetAdapter(subset, task_kind="semantic")
        return DataLoader(
            adapter,
            batch_size=self.args.batch,  # type: ignore[attr-defined]
            shuffle=False,
            collate_fn=semantic_collate_fn,
            pin_memory=self.device.type != "cpu",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
        )
