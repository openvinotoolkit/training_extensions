# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base trainer mixin with shared logic for the getitune DataModule bridge."""

from __future__ import annotations

import csv
import logging
import multiprocessing
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from ultralytics.data.build import InfiniteDataLoader, seed_worker

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import detection_collate_fn

if TYPE_CHECKING:
    from getitune.data.module import DataModule

logger = logging.getLogger(__name__)

_MP_CONTEXT = multiprocessing.get_context("spawn")

# Ultralytics ships bundled third-party experiment-logger callbacks that are
# auto-registered in ``BaseTrainer.__init__`` via ``add_integration_callbacks``.
# getitune owns experiment tracking itself (CSV metrics, and MLflow via the
# benchmark runner / application backend), so these would double-log: they
# create duplicate runs (one per work-dir / seed), may fail artifact uploads,
# and — for server-backed loggers like MLflow — can block the training process
# on the remote tracking server. They are stripped whenever the getitune
# DataModule bridge is active.
_EXTERNAL_LOGGER_CALLBACK_MODULES = frozenset(
    {
        "ultralytics.utils.callbacks.mlflow",
        "ultralytics.utils.callbacks.clearml",
        "ultralytics.utils.callbacks.comet",
        "ultralytics.utils.callbacks.dvc",
        "ultralytics.utils.callbacks.neptune",
        "ultralytics.utils.callbacks.wb",
        "ultralytics.utils.callbacks.tensorboard",
    },
)


class GetiTuneBaseTrainer:
    """Shared DataModule bridge logic for Ultralytics trainers."""

    _datamodule: DataModule | None = None
    _use_getitune_data: bool = False
    _task_kind: ClassVar[str] = "detect"
    _collate_fn = staticmethod(detection_collate_fn)
    _progress_fn: Any = None
    _progress_min: float = 0.0
    _progress_max: float = 100.0

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the trainer and disable bundled third-party loggers.

        Ultralytics registers its integration logger callbacks (MLflow, W&B,
        Comet, ClearML, Neptune, TensorBoard) inside ``BaseTrainer.__init__``.
        Once the base initializer has run, strip those callbacks when the
        getitune DataModule bridge is active so only getitune-managed logging
        remains.
        """
        super().__init__(*args, **kwargs)
        if self._use_getitune_data:
            self._disable_external_logger_callbacks()

    def _disable_external_logger_callbacks(self) -> None:
        """Remove Ultralytics' bundled third-party experiment-logger callbacks."""
        callbacks = getattr(self, "callbacks", None)
        if not isinstance(callbacks, dict):
            return
        removed = 0
        for event, fns in callbacks.items():
            kept = [fn for fn in fns if getattr(fn, "__module__", "") not in _EXTERNAL_LOGGER_CALLBACK_MODULES]
            removed += len(fns) - len(kept)
            callbacks[event] = kept
        if removed:
            logger.info(
                f"Disabled {removed} Ultralytics third-party logger callback(s); getitune manages experiment tracking."
            )

    def get_dataset(self) -> dict[str, Any]:
        """Build data config dict from DataModule or fall back to YAML."""
        if not self._use_getitune_data:
            return super().get_dataset()  # type: ignore[misc]

        if self._datamodule is None:
            msg = "DataModule is required when _use_getitune_data=True"
            raise RuntimeError(msg)

        li = self._datamodule.label_info  # type: ignore[union-attr]
        names = dict(enumerate(li.label_names))
        return {
            "train": "datamodule://train",
            "val": "datamodule://val",
            "nc": li.num_classes,
            "names": names,
            "channels": 3,
        }

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> UltralyticsDatasetAdapter:
        """Return adapter wrapping the appropriate DataModule subset.

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
        return UltralyticsDatasetAdapter(vision_dataset, task_kind=self._task_kind)

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> DataLoader:
        """Build a DataLoader from the adapter dataset."""
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
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=False,
            multiprocessing_context=_MP_CONTEXT if nw > 0 else None,
            persistent_workers=nw > 0,
            worker_init_fn=seed_worker,
        )

    def _setup_train(self) -> None:
        """Restore workers, run parent setup, then fix warmup for small datasets.

        Ultralytics 8.4+ sets ``self.args.workers = 0`` when the device is
        CPU (``if self.device.type in {"cpu", "mps"}``) during ``__init__``.
        For the DataModule bridge this is counter-productive — our
        CPU-augmentation pipeline (CachedMosaic, colour jitter, etc.) runs
        in the DataLoader workers and benefits from parallelism.  We
        restore a sensible default before the parent ``_setup_train``
        creates the dataloaders.

        Ultralytics enforces a minimum of 100 warmup iterations regardless
        of dataset size (``max(round(warmup_epochs * nb), 100)``).  For
        small datasets this can make warmup consume an unreasonable fraction
        of total training (e.g. 33+ epochs of warmup with only 3 batches/epoch).

        When the natural warmup (``warmup_epochs * nb``) is below 100 we
        disable the built-in warmup entirely and register a custom
        ``on_train_batch_start`` callback that applies the same LR /
        momentum ramp but respects the natural iteration count.
        """
        if self._use_getitune_data and self.args.workers == 0:  # type: ignore[attr-defined]
            self.args.workers = 4  # type: ignore[attr-defined]
        super()._setup_train()  # type: ignore[misc]

        if self._use_getitune_data:
            self._register_iteration_timer()

        if not self._use_getitune_data:
            return

        self._register_progress_callback()

        if self.args.warmup_epochs <= 0:  # type: ignore[attr-defined]
            return

        nb = len(self.train_loader)  # type: ignore[attr-defined]
        natural_nw = round(self.args.warmup_epochs * nb)  # type: ignore[attr-defined]
        if natural_nw < 100:
            logger.info(
                f"Bypassing Ultralytics 100-iteration warmup minimum. "
                f"With {nb} batches/epoch, using natural warmup of "
                f"{natural_nw} iterations ({self.args.warmup_epochs} epochs)."
            )

            warmup_bias_lr = self.args.warmup_bias_lr  # type: ignore[attr-defined]
            warmup_momentum = self.args.warmup_momentum  # type: ignore[attr-defined]
            nbs = self.args.nbs  # type: ignore[attr-defined]
            batch_size = self.batch_size  # type: ignore[attr-defined]
            self.args.warmup_epochs = 0  # type: ignore[attr-defined]

            counter = {"ni": 0}

            def _warmup_callback(trainer: Any) -> None:  # noqa: ANN401
                ni = counter["ni"]
                if ni <= natural_nw:
                    xi = [0, natural_nw]
                    accumulate = np.interp(ni, xi, [1.0, float(nbs / batch_size)])
                    trainer.accumulate = max(1, round(accumulate))
                    for pg in trainer.optimizer.param_groups:
                        pg["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                warmup_bias_lr if pg.get("param_group") == "bias" else 0.0,
                                float(pg["initial_lr"]) * float(trainer.lf(trainer.epoch)),
                            ],
                        )
                        if "momentum" in pg:
                            pg["momentum"] = np.interp(ni, xi, [warmup_momentum, trainer.args.momentum])
                counter["ni"] += 1

            self.add_callback("on_train_batch_start", _warmup_callback)  # type: ignore[attr-defined]

    def _register_iteration_timer(self) -> None:
        """Record per-batch train time and persist epoch means for benchmarking."""
        times_by_epoch: dict[int, list[float]] = {}
        state: dict[str, float] = {}

        def on_epoch_start(_trainer: Any) -> None:  # noqa: ANN401
            state.clear()

        def on_batch_start(_trainer: Any) -> None:  # noqa: ANN401
            state.setdefault("end", time.perf_counter())

        def on_batch_end(trainer: Any) -> None:  # noqa: ANN401
            previous_end = state.get("end")
            if previous_end is None:
                return
            current_end = time.perf_counter()
            epoch = int(getattr(trainer, "epoch", 0))
            times_by_epoch.setdefault(epoch, []).append(current_end - previous_end)
            state["end"] = current_end

        def on_train_end(trainer: Any) -> None:  # noqa: ANN401
            results_csv = Path(getattr(trainer, "save_dir", ".")) / "results.csv"
            if not results_csv.exists():
                return
            try:
                with results_csv.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream))
            except OSError:
                return
            if not rows:
                return

            header = rows[0]
            if "train/iter_time" in header:
                return
            header.append("train/iter_time")
            epoch_times = {epoch: sum(times) / len(times) for epoch, times in times_by_epoch.items() if times}
            # Ultralytics persists one-based epoch values in results.csv
            # (``self.epoch + 1``). Map by the row's epoch value, not its
            # position, so resumed runs with pre-existing rows stay aligned.
            epoch_col = header.index("epoch") if "epoch" in header else None
            for row in rows[1:]:
                epoch_key = None
                if epoch_col is not None and epoch_col < len(row):
                    try:
                        epoch_key = int(float(row[epoch_col])) - 1
                    except ValueError:
                        epoch_key = None
                row.append(str(epoch_times.get(epoch_key, "")) if epoch_key is not None else "")

            with results_csv.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(rows)

        self.add_callback("on_train_epoch_start", on_epoch_start)  # type: ignore[attr-defined]
        self.add_callback("on_train_batch_start", on_batch_start)  # type: ignore[attr-defined]
        self.add_callback("on_train_batch_end", on_batch_end)  # type: ignore[attr-defined]
        self.add_callback("on_train_end", on_train_end)  # type: ignore[attr-defined]

    def _register_progress_callback(self) -> None:
        """Register a progress-reporting callback for the training loop.

        Bridges the application's progress callable (passed via
        ``_progress_fn``) into the Ultralytics callback system.
        Computes progress as a linear interpolation between ``_progress_min``
        and ``_progress_max`` based on ``current_step / total_steps``.
        """
        if self._progress_fn is None:
            return

        progress_fn = self._progress_fn
        min_p = self._progress_min
        max_p = self._progress_max
        nb = len(self.train_loader)  # type: ignore[attr-defined]
        total_steps = max(1, self.epochs * nb)  # type: ignore[attr-defined]
        step_counter = {"step": 0}

        def _progress_callback(_trainer: Any) -> None:  # noqa: ANN401
            step_counter["step"] += 1
            ratio = step_counter["step"] / total_steps
            progress = min_p + ratio * (max_p - min_p)
            progress_fn(progress)

        self.add_callback("on_train_batch_end", _progress_callback)  # type: ignore[attr-defined]
        logger.info(f"Registered progress callback: {total_steps} total steps, range [{min_p}, {max_p}]")

    def _clear_memory(self, threshold: float | None = None) -> None:
        """Lightweight memory clearing that skips ``gc.collect()``.

        The upstream implementation calls ``gc.collect()`` every epoch which
        forces a full Python garbage-collection cycle.  This is expensive
        with large object graphs (spawn-based DataLoader workers, Datumaro
        caches, etc.) and unnecessary during tight training loops where
        memory pressure is manageable via CUDA cache management alone.

        Falls back to the upstream implementation when running without the
        DataModule bridge (native YOLO data path).
        """
        if not self._use_getitune_data:
            return super()._clear_memory(threshold)  # type: ignore[misc]

        if self.device.type == "cpu":  # type: ignore[attr-defined]
            return None

        if threshold is not None and self._get_memory(fraction=True) <= threshold:  # type: ignore[attr-defined]
            return None

        if self.device.type == "mps":  # type: ignore[attr-defined]
            torch.mps.empty_cache()
        elif self.device.type == "xpu":  # type: ignore[attr-defined]
            torch.xpu.empty_cache()
        else:
            torch.cuda.empty_cache()
        return None

    def set_model_attributes(self) -> None:
        """Set model attributes; disable Ultralytics augmentations when using DataModule."""
        super().set_model_attributes()  # type: ignore[misc]
        if self._use_getitune_data:
            self._disable_ultralytics_augmentations()

    def set_class_weights(self) -> None:
        """Skip class-weight computation (adapter has no ``labels`` attr)."""
        if self._use_getitune_data:
            return
        super().set_class_weights()  # type: ignore[misc]

    def optimizer_step(self) -> None:
        """Perform optimizer step with configurable gradient clipping.

        Ultralytics hardcodes ``max_norm=10.0``.  This override reads
        ``_max_grad_norm`` from the trainer class attribute (set by the
        engine when popping ``max_grad_norm`` from train args) so users can
        control clipping via the backend UI.  A value of ``0.0`` disables
        clipping entirely.
        """
        max_norm = getattr(self, "_max_grad_norm", None)
        if max_norm is None:
            max_norm = getattr(self.args, "max_grad_norm", 10.0)  # type: ignore[attr-defined]
        self.scaler.unscale_(self.optimizer)  # type: ignore[attr-defined]
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)  # type: ignore[attr-defined]
        self.scaler.step(self.optimizer)  # type: ignore[attr-defined]
        self.scaler.update()  # type: ignore[attr-defined]
        self.optimizer.zero_grad()  # type: ignore[attr-defined]
        if self.ema:  # type: ignore[attr-defined]
            self.ema.update(self.model)  # type: ignore[attr-defined]

    def plot_training_labels(self) -> None:
        """Skip label plotting (adapter has no ``labels`` attr)."""
        if self._use_getitune_data:
            return
        super().plot_training_labels()  # type: ignore[misc]

    def auto_batch(self) -> int:
        """Skip auto-batch when using DataModule."""
        if self._use_getitune_data:
            return self.batch_size  # type: ignore[attr-defined]
        return super().auto_batch()  # type: ignore[misc]

    def _disable_ultralytics_augmentations(self) -> None:
        """Zero out Ultralytics augmentation hyperparams.

        All augmentations are handled by the DataModule pipeline, so we
        disable all upstream augmentation parameters to prevent double
        augmentation.
        """
        for attr in (
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "flipud",
            "fliplr",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "close_mosaic",
        ):
            if hasattr(self.args, attr):  # type: ignore[attr-defined]
                setattr(self.args, attr, 0.0 if attr != "close_mosaic" else 0)  # type: ignore[attr-defined]
