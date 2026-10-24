# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base validator mixin with shared standalone evaluation logic."""

from __future__ import annotations

import multiprocessing
import time
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch
from torch.utils.data import DataLoader
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import RANK, TQDM, callbacks
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import select_device, unwrap_model

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer

    from getitune.data.module import DataModule


class GetiTuneValidatorMixin:
    """Mixin providing shared standalone evaluation logic for Ultralytics validators.

    When ``_datamodule`` is set and called without a trainer, runs
    standalone validation bypassing Ultralytics' YAML data parsing.

    Subclasses must set :attr:`_task_kind` to dispatch the adapter to the
    correct per-task ``_getitem_*`` method (e.g. ``"detect"``, ``"segment"``).

    ``_datamodule`` is bound one of two ways:

    - As a class attribute on a dynamically-created subclass (see
      ``UltralyticsEngine._make_bound_validator``), for validators that
      Ultralytics constructs internally (e.g. via ``model.val()``), where we
      don't control the call site's arguments.
    - Via :meth:`set_datamodule`, for validators we construct ourselves
      (e.g. in a trainer's ``get_validator()``).
    """

    _datamodule: DataModule | None = None
    _task_kind: ClassVar[str] = "detect"
    _collate_fn: ClassVar[Callable]

    @property
    def datamodule(self) -> DataModule | None:
        """Get the bound DataModule instance."""
        return self._datamodule

    @datamodule.setter
    def datamodule(self, datamodule: DataModule | None) -> None:
        """Bind a DataModule instance, overriding any class-level default."""
        self._datamodule = datamodule

    def __call__(self, trainer: BaseTrainer | None = None, model: torch.nn.Module | None = None) -> dict:
        """Dispatch to upstream (training) or standalone DataModule validation."""
        if trainer is not None or self._datamodule is None:
            return super().__call__(trainer=trainer, model=model)  # type: ignore[misc]
        return self._run_standalone_eval(model)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensors to device, tracking per-batch wall time for ``iter_time`` metrics."""
        now = time.perf_counter()
        batch_times = getattr(self, "_batch_times", None)
        if batch_times is None:
            batch_times = []
            self._batch_times = batch_times
        batch_start = getattr(self, "_batch_start", None)
        if batch_start is not None:
            batch_times.append(now - batch_start)
        self._batch_start = now

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)  # type: ignore[attr-defined]
        if getattr(self.args, "quantize", None) == 16:  # type: ignore[attr-defined]
            batch["img"] = batch["img"].half()
        if "masks" in batch and isinstance(batch["masks"], torch.Tensor):
            batch["masks"] = batch["masks"].float()
        return batch

    def _average_iter_time(self) -> float:
        """Trimmed-mean per-batch wall time, excluding the first (warmup) batch.

        Mirrors ``UltralyticsEngine._summarize_iter_times``: the first batch
        includes one-off warmup costs (CUDA/XPU kernel compilation, first-call
        overhead) that would otherwise skew the average.
        """
        batch_times: list[float] = getattr(self, "_batch_times", None) or []
        if not batch_times:
            return 0.0
        trimmed = batch_times[1:] if len(batch_times) > 1 else batch_times
        return sum(trimmed) / len(trimmed)

    def _run_standalone_eval(self, model: torch.nn.Module | None) -> dict:
        """Run validation using DataModule without YAML data config.

        Replicates the essential non-training setup from
        ``BaseValidator.__call__`` but skips ``check_det_dataset()``
        which requires a YAML file on disk.
        """
        self.training = False  # type: ignore[attr-defined]
        callbacks.add_integration_callbacks(self)

        raw_dev = self.args.device  # type: ignore[attr-defined]
        if isinstance(raw_dev, torch.device) and raw_dev.type == "xpu":
            dev = raw_dev
        elif isinstance(raw_dev, str) and "xpu" in raw_dev:
            dev = torch.device(raw_dev)
        else:
            dev = select_device(raw_dev) if RANK == -1 else torch.device("cuda", RANK)
        model = AutoBackend(
            model=model or self.args.model,  # pyrefly: ignore[bad-argument-type]  # type: ignore[attr-defined]
            device=dev,  # pyrefly: ignore[bad-argument-type]
            fp16=getattr(self.args, "quantize", None) == 16,  # type: ignore[attr-defined]
        )
        self.device = model.device  # type: ignore[attr-defined]
        self.args.quantize = 16 if model.fp16 else None  # type: ignore[attr-defined]
        imgsz = check_imgsz(self.args.imgsz, stride=model.stride)  # type: ignore[attr-defined]

        if self._datamodule is None:
            msg = "Standalone evaluation requires a DataModule"
            raise TypeError(msg)
        li = self._datamodule.label_info
        self.data = {  # type: ignore[attr-defined]
            "nc": li.num_classes,
            "names": dict(enumerate(li.label_names)),
            "channels": 3,
            "val": "",
        }

        self.dataloader = self.dataloader or self._build_adapter_dataloader()  # type: ignore[attr-defined]

        model.eval()
        if isinstance(imgsz, int):
            warmup_imgsz = (1, int(self.data["channels"]), imgsz, imgsz)
        else:
            warmup_imgsz = (1, int(self.data["channels"]), int(imgsz[0]), int(imgsz[1]))
        model.warmup(imgsz=warmup_imgsz)  # type: ignore[attr-defined]

        self.run_callbacks("on_val_start")  # type: ignore[attr-defined]
        dt = tuple(Profile(device=self.device) for _ in range(4))  # type: ignore[attr-defined]
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))  # type: ignore[attr-defined]
        self.init_metrics(unwrap_model(model))  # type: ignore[attr-defined]
        self.jdict = []  # type: ignore[attr-defined]

        for batch_i, raw_batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")  # type: ignore[attr-defined]
            self.batch_i = batch_i  # type: ignore[attr-defined]
            with dt[0]:
                batch = self.preprocess(raw_batch)
            with dt[1]:
                preds = model(batch["img"])
            with dt[2]:
                pass  # no loss in standalone mode
            with dt[3]:
                preds = self.postprocess(preds)  # type: ignore[attr-defined]
            self.update_metrics(preds, batch)  # type: ignore[attr-defined]
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:  # type: ignore[attr-defined]
                self.plot_val_samples(batch, batch_i)  # type: ignore[attr-defined]
                self.plot_predictions(batch, preds, batch_i)  # type: ignore[attr-defined]
            self.run_callbacks("on_val_batch_end")  # type: ignore[attr-defined]

        stats: dict = {}
        self.gather_stats()  # type: ignore[attr-defined]
        if RANK in {-1, 0}:
            stats = self.get_stats()  # type: ignore[attr-defined]
            ds_len = len(self.dataloader.dataset)  # pyrefly: ignore[bad-argument-type]  # type: ignore[attr-defined]
            self.speed = dict(zip(self.speed.keys(), (x.t / ds_len * 1e3 for x in dt)))  # type: ignore[attr-defined]
            self.finalize_metrics()  # type: ignore[attr-defined]
            self.print_results()  # type: ignore[attr-defined]
            self.run_callbacks("on_val_end")  # type: ignore[attr-defined]
        return stats

    def _build_adapter_dataloader(self) -> DataLoader:
        """Build a DataLoader from the DataModule's val/test subset."""
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
            collate_fn=self._collate_fn,
            pin_memory=self.device.type != "cpu",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
        )
