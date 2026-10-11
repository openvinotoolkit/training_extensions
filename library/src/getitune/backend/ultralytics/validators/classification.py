# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Classification validators for the getitune data bridge."""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from torchmetrics import Metric, MetricCollection
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score
from ultralytics.models.yolo.classify import ClassificationValidator as _UltralyticsClassificationValidator
from ultralytics.utils import LOGGER

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import classification_collate_fn, multilabel_collate_fn
from getitune.metrics.accuracy import MultiLabelClsMetricCallable
from getitune.types.label import LabelInfo

from .base import GetiTuneValidatorMixin


class ClassificationValidator(GetiTuneValidatorMixin, _UltralyticsClassificationValidator):
    """Classification validator for the getitune data bridge."""

    _task_kind: ClassVar[str] = "classify"

    def get_stats(self) -> dict[str, float]:
        """Add macro F1-score and iteration time to the upstream accuracy stats.

        Upstream ``ClassificationValidator.get_stats`` only reports top-1/top-5
        accuracy. ``self.pred``/``self.targets`` (accumulated by the upstream
        ``update_metrics``) already hold everything needed for F1: the first
        column of each top-N prediction tensor is the top-1 predicted class.
        """
        stats = super().get_stats()  # type: ignore[misc]
        if self.pred and self.targets:  # type: ignore[attr-defined]
            preds_top1 = torch.cat([p[:, 0] for p in self.pred])  # type: ignore[attr-defined]
            targets_all = torch.cat(self.targets)  # type: ignore[attr-defined]
            f1_metric = MulticlassF1Score(num_classes=self.nc, average="macro")  # type: ignore[attr-defined]
            stats["metrics/f1-score"] = float(f1_metric(preds_top1, targets_all))
        stats["metrics/iter_time"] = self._average_iter_time()
        return stats

    def _build_adapter_dataloader(self) -> DataLoader:
        """Build a classification DataLoader from the DataModule's val/test subset."""
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
        adapter = UltralyticsDatasetAdapter(subset, task_kind=self._task_kind)
        return DataLoader(
            adapter,
            batch_size=self.args.batch,  # type: ignore[attr-defined]
            shuffle=False,
            collate_fn=classification_collate_fn,
            pin_memory=self.device.type != "cpu",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
        )


class MultiLabelClassificationValidator(GetiTuneValidatorMixin, _UltralyticsClassificationValidator):
    """Multi-label classification validator using getitune torchmetrics."""

    _task_kind: ClassVar[str] = "multilabel"

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize multi-label metrics from DataModule label info."""
        raw_names = getattr(model, "names", None)
        names: dict[int, str] = raw_names if isinstance(raw_names, dict) else {}
        self.names = names
        self.nc = len(names)
        label_info = self._label_info_from_names(names)
        self.metric: Metric | MetricCollection = MultiLabelClsMetricCallable(label_info)
        self.metric.to(self.device)  # type: ignore[attr-defined]

    def _label_info_from_names(self, names: dict[int, str]) -> LabelInfo:
        """Build a ``LabelInfo`` with per-label groups for multi-label metrics."""
        if self._datamodule is not None:
            label_info = self._datamodule.label_info
            if all(len(group) == 1 for group in label_info.label_groups):
                return label_info
            return LabelInfo(
                label_names=label_info.label_names,
                label_groups=[[name] for name in label_info.label_names],
                label_ids=label_info.label_ids,
            )
        name_list = list(names.values())
        return LabelInfo(
            label_names=name_list,
            label_groups=[[name] for name in name_list],
            label_ids=[str(i) for i in range(len(name_list))],
        )

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """Accumulate sigmoid predictions and multi-hot targets."""
        if getattr(self, "metric", None) is None:
            msg = "Metric is not initialized; call init_metrics() before update_metrics()."
            raise RuntimeError(msg)
        target = batch["cls"].float()
        self.metric.update(preds=preds, target=target)

    def finalize_metrics(self) -> None:
        """No-op: metric is computed in ``get_stats``."""

    def gather_stats(self) -> None:
        """No-op: torchmetrics state is kept on the local rank."""

    def get_stats(self) -> dict[str, float]:
        """Compute and return multi-label accuracy, mAP, and iteration time."""
        if getattr(self, "metric", None) is None:
            msg = "Metric is not initialized; call init_metrics() before get_stats()."
            raise RuntimeError(msg)
        results = self.metric.compute()
        accuracy = self._extract_scalar(results, "accuracy")
        mean_ap = self._extract_scalar(results, "map")
        return {
            "metrics/accuracy": accuracy,
            "metrics/map": mean_ap,
            "metrics/iter_time": self._average_iter_time(),
        }

    @staticmethod
    def _extract_scalar(results: dict[str, Any], key: str) -> float:
        """Extract a scalar value from a possibly-nested metric result dict."""
        value = results.get(key)
        if isinstance(value, dict):
            value = value.get(key, value.get("accuracy"))
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    def print_results(self) -> None:
        """Print multi-label validation metrics."""
        stats = self.get_stats()
        LOGGER.info(f"{'multi-label accuracy':>20}: {stats['metrics/accuracy']:.3g}")
        LOGGER.info(f"{'multi-label mAP':>20}: {stats['metrics/map']:.3g}")

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Skip image plotting for multi-label batches.

        Upstream ``ClassificationValidator.plot_val_samples``/``plot_predictions``
        assume a single scalar class id per image (``batch["cls"]`` of shape
        ``(N,)``). Multi-label ``cls`` is a multi-hot vector of shape
        ``(N, num_labels)``, which crashes Ultralytics' ``plot_images`` /
        ``colors()`` (expects a 0-d class id). There is no established
        single-image rendering for multiple simultaneous tags, so plotting
        is skipped rather than producing a misleading or crashing visualization.
        """

    def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None:
        """Skip prediction plotting for multi-label batches. See :meth:`plot_val_samples`."""

    def _build_adapter_dataloader(self) -> DataLoader:
        """Build a multi-label DataLoader from the DataModule's val/test subset."""
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
        adapter = UltralyticsDatasetAdapter(subset, task_kind=self._task_kind)
        return DataLoader(
            adapter,
            batch_size=self.args.batch,  # type: ignore[attr-defined]
            shuffle=False,
            collate_fn=multilabel_collate_fn,
            pin_memory=self.device.type != "cpu",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
        )
