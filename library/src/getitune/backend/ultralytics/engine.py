# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics engine implementation."""

from __future__ import annotations

import csv
import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Sequence

import torch
import torch.nn.functional as f
import yaml
from torchvision import tv_tensors
from ultralytics.utils.metrics import ClassifyMetrics, DetMetrics, SegmentMetrics

from getitune.backend.ultralytics.data.geometry import scale_boxes_to_letterbox, scale_masks_to_letterbox
from getitune.backend.ultralytics.tools.configurator import Configurator
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import Prediction, SampleBatch
from getitune.data.module import DataModule
from getitune.data.utils.structures.mask.mask_util import encode_rle
from getitune.engine.engine import Engine
from getitune.metrics.accuracy import MultiLabelClsMetricCallable
from getitune.metrics.dice import SegmCallable
from getitune.metrics.fmeasure import FMeasure
from getitune.types.device import DeviceType
from getitune.types.export import ExportFormat
from getitune.types.label import SegLabelInfo
from getitune.types.precision import Precision
from getitune.utils.device import is_xpu_available

from .models.base import UltralyticsModel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lightning_fabric.plugins.precision.precision import _PRECISION_INPUT
    from torch.utils.data import DataLoader
    from torchmetrics import Metric, MetricCollection
    from ultralytics import YOLO

    from getitune.types import PathLike
    from getitune.types.label import LabelInfo
    from getitune.types.types import ANNOTATIONS, DATA, METRICS, MODEL

logger = logging.getLogger(__name__)

# Unified type for ultralytics metrics objects returned by train/val
UMETRICS = DetMetrics | SegmentMetrics | ClassifyMetrics


class _UltralyticsResultLike(Protocol):
    orig_img: Any
    orig_shape: tuple[int, int]


class UltralyticsEngine(Engine):
    """Engine backed by ``ultralytics.YOLO``.

    Wraps an :class:`UltralyticsModel` and a
    :class:`~getitune.data.module.DataModule` (or data-root path).
    """

    _EXPORTED_MODEL_BASE_NAME: ClassVar[str] = "exported_model"
    _LAST_TRAIN_CHECKPOINT_FILE: ClassVar[str] = ".last_train_checkpoint"

    def __init__(
        self,
        model: UltralyticsModel,
        data: DataModule | PathLike,
        work_dir: PathLike | None = None,
        device: str | DeviceType = "auto",
        checkpoint: PathLike | None = None,
        train_args: Mapping[str, Any] | None = None,
        export_args: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the engine.

        Args:
            model: Ultralytics model wrapper.
            data: DataModule or filesystem data-root path.
            work_dir: Directory for checkpoints, exports, and logs.
            device: Device string or :class:`DeviceType` enum
                (``"auto"``, ``"xpu"``, ``"0"``, ``"cpu"``, ``DeviceType.xpu``, etc.).
            checkpoint: Optional path to a checkpoint (.pt) to load model
                weights from before training.  Used for both pretrained
                base weights and parent-revision warm-start.
            train_args: Train-only defaults forwarded to ``yolo.train()``.
            export_args: Export metadata defaults, such as inference thresholds.
            **kwargs: Extra overrides forwarded to Ultralytics calls.
        """
        if not isinstance(model, UltralyticsModel):
            msg = f"model must be an UltralyticsModel instance, got {type(model)}"
            raise TypeError(msg)

        self._model = model
        self._work_dir = Path(work_dir or "./getitune-workspace").resolve()
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._device = self._resolve_device(device)
        self._kwargs = kwargs
        self._train_args = dict(train_args or {})
        self._export_args = dict(export_args or {})
        self._last_train_checkpoint = self._load_last_train_checkpoint()

        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)

        if isinstance(data, DataModule):
            self._datamodule: DataModule | None = data
            self._data_root: Path | None = None
        elif isinstance(data, (str, os.PathLike)):
            self._datamodule = None
            self._data_root = Path(data)
        else:
            msg = f"data must be DataModule or PathLike, got {type(data)}"
            raise TypeError(msg)

        # Propagate intensity config from DataModule so the exporter can embed
        # the correct input_dtype and intensity_mode into the exported model.
        if self._datamodule is not None:
            intensity_cfg = getattr(self._datamodule, "input_intensity_config", None)
            if intensity_cfg is not None:
                self._model.set_intensity_config(intensity_cfg)

    def train(
        self,
        max_epochs: int | None = None,
        batch: int | None = None,
        lr0: float | None = None,
        patience: int | None = None,
        precision: _PRECISION_INPUT | None = "16-mixed",
        callbacks: list[Any] | None = None,
        **kwargs,
    ) -> METRICS:
        """Train the model via a custom Ultralytics trainer.

        Args:
            max_epochs: Number of training epochs (canonical cross-backend name).
            batch: Batch size.
            lr0: Initial learning rate.
            patience: Early stopping patience (0 to disable).
            precision: Training precision.  Accepted values: ``"16-mixed"``, ``"16"``, ``"bf16-mixed"``,
                ``"bf16"`` (mixed precision / AMP), ``"32"``, ``"32-true"``
                (full FP32).  ``None`` leaves the Ultralytics default
                (``amp=True``, i.e. FP16).  On CPU any FP16/BF16 variant is
                downgraded to FP32 (a warning is logged).  On XPU the mixin
                converts the model to BF16 regardless of whether ``"16"`` or
                ``"bf16"`` was requested.
            callbacks: Accepted for API compatibility; unused by Ultralytics.
            **kwargs: Additional overrides forwarded to Ultralytics training.

        Returns:
            Translated metric dict.
        """
        progress_fn, progress_min, progress_max = self._extract_progress_callback(callbacks)
        # Translate Lightning-specific 'devices' to Ultralytics device selection.
        if "devices" in kwargs:
            devices = kwargs.pop("devices")
            if isinstance(devices, list) and len(devices) >= 1:
                idx = devices[0]
                if len(devices) > 1:
                    logger.warning(f"UltralyticsEngine does not support multi-device; using first device: {idx}")
                # Only set index for accelerator devices; CPU doesn't support indices.
                dev_type = self._device.type
                if dev_type != "cpu":
                    self._device = torch.device(f"{dev_type}:{idx}")
        explicit: dict[str, Any] = {}
        if max_epochs is not None:
            explicit["epochs"] = max_epochs
        if batch is not None:
            explicit["batch"] = batch
        if lr0 is not None:
            explicit["lr0"] = lr0
        if patience is not None:
            explicit["patience"] = patience
        kwargs.update(explicit)
        yolo = self._model.yolo
        merged = self._build_overrides(self._train_args, **kwargs)

        if self._data_root is not None and "data" not in merged:
            merged["data"] = str(self._data_root)

        # Pop before yolo.train() — Ultralytics rejects unknown keys.
        max_grad_norm = merged.pop("max_grad_norm", None)

        trainer_cls = self._make_bound_trainer(
            progress_fn=progress_fn,
            progress_min=progress_min,
            progress_max=progress_max,
            max_grad_norm=max_grad_norm,
        )
        train_args = {
            "trainer": trainer_cls,
            "device": self._device,
            "imgsz": self._model.imgsz,
            "project": str(self._work_dir),
            "name": "train",
            "exist_ok": True,
            **merged,
        }

        # Inject amp after **merged so it wins over recipe values.
        amp_val = self._precision_to_amp(precision, self._device)
        if amp_val is not None:
            train_args["amp"] = amp_val

        logger.info(
            f"Starting Ultralytics training: model={self._model.model_name}, "
            f"device={self._device}, imgsz={self._model.imgsz}"
        )

        results = yolo.train(**train_args)
        self._record_last_train_checkpoint(self._resolve_trainer_checkpoint(yolo))
        self._remap_results_csv()
        # Confidence-threshold tuning only applies to box/mask tasks.
        # Classification has no box predictions and therefore no FMeasure sweep.
        if self._model.task in ("detect", "segment"):
            best_val_confidence_thr = self._compute_best_confidence_threshold()
            if best_val_confidence_thr is not None:
                # Cap at the user-specified confidence threshold (0.25 by default)
                # to avoid overly aggressive pruning that can harm export performance.
                threshold = min(best_val_confidence_thr, self._export_args.get("confidence_threshold", 0.25))
                self._export_args["confidence_threshold"] = threshold
                logger.info(f"Best confidence threshold from FMeasure: {threshold:.4f}")
        return self._translate_metrics(results)

    def test(self, checkpoint: PathLike | None = None, metric: Callable[..., Any] | None = None, **kwargs) -> METRICS:
        """Evaluate the model using torchmetrics or the Ultralytics validator.

        When a ``metric`` callable is provided **and** a DataModule is
        attached, evaluation uses the same torchmetrics pipeline as
        Lightning (e.g. ``MeanAveragePrecision``).  This ensures metric
        consistency across backends.

        When ``metric`` is ``None``, falls back to the Ultralytics
        built-in validator (``DetMetrics``).

        Args:
            checkpoint: Optional ``.pt`` checkpoint to evaluate.
            metric: A ``MetricCallable`` — a function that accepts
                ``LabelInfo`` and returns a ``torchmetrics.Metric``.
                When provided, the torchmetrics evaluation path is used.
            **kwargs: Overrides forwarded to validation.

        Returns:
            Metric dict.  Keys are prefixed with ``test/`` when using
            torchmetrics, or ``val/`` when using the YOLO validator.
        """
        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)

        if metric is not None and self._datamodule is not None:
            return self._test_with_torchmetrics(metric)

        if self._datamodule is not None and getattr(self._model, "is_multilabel", False):
            return self._test_with_torchmetrics(MultiLabelClsMetricCallable)

        if self._datamodule is not None and self._model.task == "semantic":
            return self._test_with_torchmetrics(SegmCallable)  # pyrefly: ignore[bad-argument-type]

        merged = self._build_overrides(**kwargs)

        if self._datamodule is not None:
            return self._test_with_datamodule(merged, checkpoint=None)

        yolo = self._model.yolo
        if self._data_root is not None and "data" not in merged:
            merged["data"] = str(self._data_root)

        val_args = {
            "device": self._device,
            "imgsz": self._model.imgsz,
            "project": str(self._work_dir),
            "name": "val",
            "exist_ok": True,
            **merged,
        }

        logger.info(f"Starting Ultralytics validation: model={self._model.model_name}")

        results = yolo.val(**val_args)
        return self._translate_metrics(results)

    def predict(
        self, source: str | Path | None = None, conf: float | None = None, iou: float | None = None, **kwargs
    ) -> ANNOTATIONS:
        """Run inference and return a list of :class:`Prediction` objects.

        When a DataModule is attached, iterates ``predict_dataloader()``.
        Otherwise uses ``yolo.predict(source=...)``.

        Args:
            source: Image source path or directory. Overrides attached data.
            conf: Confidence threshold for predictions.
            iou: IoU threshold for NMS.
            **kwargs: Additional overrides forwarded to prediction.
        """
        extra: dict[str, Any] = {}
        if source is not None:
            extra["source"] = str(source)
        if conf is not None:
            extra["conf"] = conf
        if iou is not None:
            extra["iou"] = iou
        merged = self._build_overrides(**extra, **kwargs)

        if self._datamodule is not None and "source" not in merged:
            return self._predict_with_datamodule(merged)  # pyrefly: ignore[bad-return]

        yolo = self._model.yolo
        resolved_source = str(merged.pop("source")) if "source" in merged else None
        if resolved_source is None and self._data_root is not None:
            resolved_source = str(self._data_root)

        predict_args = {
            "source": resolved_source,
            "device": self._device,
            "imgsz": self._model.imgsz,
            "project": str(self._work_dir),
            "name": "predict",
            "exist_ok": True,
            "save": False,
            **merged,
        }

        self._model.ensure_predict_ready()

        raw_results = yolo.predict(**predict_args)  # pyrefly: ignore[bad-argument-type]

        return self._convert_predictions(raw_results)  # pyrefly: ignore[bad-return]

    def export(
        self,
        checkpoint: PathLike | None = None,
        export_format: ExportFormat = ExportFormat.OPENVINO,
        export_precision: Precision = Precision.FP32,
        export_nms: bool = False,
        **kwargs,
    ) -> Path:
        """Export the model to OpenVINO IR or ONNX.

        Delegates to :meth:`UltralyticsModel.export` which follows the same
        architecture as Lightning — metadata embedding, preprocessing
        parameters, and FP16 compression are handled by the model's exporter.

        Args:
            checkpoint: Path to weights to export. When given, the model loads
                weights from this file before exporting.
            export_format: Target format.
            export_precision: Precision (FP32 or FP16).
            export_nms: Whether to include NMS in the exported model graph.
                Defaults to False.
            **kwargs: Extra arguments (reserved for future use).

        Returns:
            Path to the exported model file (``.xml`` for OpenVINO,
            ``.onnx`` for ONNX).
        """
        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)
        elif self._last_train_checkpoint is not None and self._last_train_checkpoint.exists():
            self._model.load_checkpoint(self._last_train_checkpoint)
        else:
            best_pt = self._work_dir / "train" / "weights" / "best.pt"
            if best_pt.exists():
                self._model.load_checkpoint(best_pt)

        logger.info(
            f"Exporting model: format={export_format.value}, "
            f"precision={export_precision.value}, "
            f"checkpoint={checkpoint or self._last_train_checkpoint or 'current weights'}"
        )

        previous_export_nms = self._model.export_nms
        self._model.export_nms = export_nms
        try:
            return self._model.export(
                output_dir=self._work_dir,
                base_name=self._EXPORTED_MODEL_BASE_NAME,
                export_format=export_format,
                precision=export_precision,
                export_args=self._export_args,
            )
        finally:
            self._model.export_nms = previous_export_nms

    @staticmethod
    def is_supported(model: MODEL, data: DATA) -> bool:
        """Return ``True`` when *model* is an :class:`UltralyticsModel`."""
        return bool(isinstance(model, UltralyticsModel) and isinstance(data, (DataModule, str, os.PathLike)))

    @classmethod
    def from_config(
        cls,
        config_path: PathLike,
        data: DataModule | PathLike | None = None,
        work_dir: PathLike | None = None,
        device: str | None = None,
        checkpoint: str | None = None,
        task: str | None = None,  # noqa: ARG003 (API compatibility with Engine.from_config)
        **kwargs,
    ) -> UltralyticsEngine:
        """Build an engine from an Ultralytics recipe configuration file.

        Args:
            config_path: Path to the Ultralytics recipe YAML file
                (must contain ``backend: ultralytics``).
            data: A pre-built :class:`~getitune.data.module.DataModule` or a
                filesystem data-root path.  Required.
            work_dir: Working directory for checkpoints and exports.
                Defaults to ``"./getitune-workspace"``.
            device: Device to use (e.g., ``"auto"``, ``"xpu"``, ``"cpu"``, ``"gpu"``).
                Defaults to None.
            checkpoint: Optional path to a checkpoint for pretrained or warm-start weights.
                Defaults to None.
            task: Task type for disambiguation when a model name matches recipes
                under multiple tasks. Not forwarded to the engine. Defaults to None.
            **kwargs: Backend-specific keyword arguments forwarded to
                :class:`UltralyticsEngine` (e.g. ``train_args``, ``export_args``).

        Returns:
            A fully configured :class:`UltralyticsEngine`.

        Raises:
            ValueError: If *data* is ``None``.
        """
        if data is None:
            msg = "data (a DataModule or data-root path) is required for UltralyticsEngine.from_config."
            raise ValueError(msg)

        # Read the task from the raw YAML (top-level field, no interpolation needed).
        with Path(str(config_path)).open() as fh:
            raw = yaml.safe_load(fh)
        recipe_task: str | None = (raw or {}).get("task")

        configurator = Configurator(data=data, model=Path(str(config_path)), task=recipe_task)

        # Build (or reuse) the DataModule, then derive label_info for the model.
        datamodule = configurator.build_datamodule()
        label_info = datamodule.label_info
        model = configurator.create_model(label_info)

        engine_kwargs: dict[str, Any] = {**kwargs}
        if device is not None:
            engine_kwargs["device"] = device
        if checkpoint is not None:
            engine_kwargs["checkpoint"] = checkpoint

        return configurator.create_engine(
            model=model,
            data=datamodule,
            work_dir=work_dir,
            **engine_kwargs,
        )

    @property
    def work_dir(self) -> PathLike:
        """Working directory."""
        return self._work_dir

    @property
    def model(self) -> UltralyticsModel:
        """The wrapped :class:`UltralyticsModel`."""
        return self._model

    @property
    def best_checkpoint(self) -> Path | None:
        """Path to the best model checkpoint after training.

        Resolution order:
        1. Recorded checkpoint from the most recent ``train()`` call.
        2. Canonical ``best_checkpoint.pt`` in the work directory.
        3. ``None`` if no checkpoint is available.
        """
        if self._last_train_checkpoint is not None and self._last_train_checkpoint.exists():
            return self._last_train_checkpoint
        canonical = self._work_dir / "best_checkpoint.pt"
        return canonical if canonical.exists() else None

    @property
    def datamodule(self) -> DATA:
        """The attached DataModule, or data root path."""
        if self._datamodule is not None:
            return self._datamodule
        if self._data_root is not None:
            return self._data_root
        msg = "No DataModule or data_root configured"
        raise ValueError(msg)

    def _test_with_datamodule(self, overrides: dict, checkpoint: PathLike | None = None) -> dict[str, float]:
        """Run validation via a bound validator class with DataModule data."""
        validator_cls = self._make_bound_validator()

        args = {
            "imgsz": self._model.imgsz,
            "device": self._device,
            "project": str(self._work_dir),
            "name": "val",
            "exist_ok": True,
            **overrides,
            "mode": "val",
        }

        logger.info(f"Starting DataModule validation: model={self._model.model_name}")

        validator = validator_cls(
            save_dir=self._work_dir / "val",
            args=args,
        )

        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)
        results = validator(model=self._model.yolo.model)
        return self._translate_metrics(results)

    def _test_with_torchmetrics(
        self,
        metric_callable: Callable[[LabelInfo], Metric | MetricCollection],
    ) -> dict[str, float]:
        """Evaluate using torchmetrics — same metrics as Lightning models.

        Iterates the DataModule's test dataloader, runs YOLO predictions on
        each batch, converts outputs to the torchmetrics dict format, and
        computes the metric.  Returns a flat dict with ``test/`` prefixed keys
        identical to those produced by ``LightningEngine.test()``.

        Supports both detection and instance segmentation:

        - **Detection**: predictions contain ``boxes``, ``scores``, ``labels``.
        - **Instance segmentation**: predictions additionally contain ``masks``
          as RLE-encoded dicts, matching ``MaskRLEMeanAveragePrecision`` format.

        Target bounding boxes and masks from the DataModule are in original
        image coordinates (``resize_targets=False``), while YOLO predictions
        are in the letterbox-padded model input space.  This method transforms
        both boxes and masks into the prediction coordinate space before
        metric update.

        Args:
            metric_callable: A function ``(LabelInfo) -> Metric``.

        Returns:
            Flat metric dict, e.g. ``{"test/map": 0.75, "test/map_50": 0.90}``.
        """
        if self._datamodule is None:
            msg = "_test_with_torchmetrics requires a DataModule"
            raise TypeError(msg)

        label_info = self._model.label_info or self._datamodule.label_info
        metric = metric_callable(label_info)
        device = self._device

        yolo = self._model.yolo
        yolo.model.to(device).eval()  # pyrefly: ignore[missing-attribute]
        metric = metric.to(device)

        dataloader = self._datamodule.test_dataloader()
        imgsz = self._model.imgsz

        logger.info(
            f"Starting torchmetrics evaluation: model={self._model.model_name}, "
            f"metric={type(metric).__name__}, batches={len(dataloader)}"
        )

        if self._model.task == "classify":
            return self._test_classification_with_torchmetrics(metric_callable, dataloader, yolo, device, imgsz)

        if self._model.task == "semantic":
            return self._test_semantic_with_torchmetrics(metric_callable, dataloader, yolo, device)

        iter_times: list[float] = []
        batch_start = time.perf_counter()
        for batch in dataloader:
            if not isinstance(batch, SampleBatch):
                msg = f"Expected test_dataloader to yield SampleBatch, got {type(batch)}"
                raise TypeError(msg)

            imgs = batch.images.to(device) if isinstance(batch.images, torch.Tensor) else batch.images
            raw_results = yolo.predict(
                source=imgs,
                device=device,
                imgsz=imgsz,
                conf=0.0,
                save=False,
                verbose=False,
            )

            preds_list = []
            for result in raw_results:
                pred_dict: dict[str, Any] = {
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                }
                if result.boxes is not None and len(result.boxes):
                    pred_dict["boxes"] = result.boxes.xyxy.to(device)  # pyrefly: ignore[missing-attribute]
                    pred_dict["scores"] = result.boxes.conf.to(device).float()  # pyrefly: ignore[missing-attribute]
                    pred_dict["labels"] = result.boxes.cls.to(device).long()  # pyrefly: ignore[missing-attribute]
                if result.masks is not None and len(result.masks):
                    masks_data = result.masks.data  # pyrefly: ignore[missing-attribute]
                    pred_dict["masks"] = [encode_rle((m > 0.5).cpu()) for m in masks_data]
                preds_list.append(pred_dict)

            target_list = []
            for i in range(len(raw_results)):
                tgt_dict: dict[str, Any] = {
                    "boxes": torch.zeros((0, 4), device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                }
                if batch.bboxes is not None and i < len(batch.bboxes):
                    boxes = batch.bboxes[i].data.to(device).float()
                    ori_h, ori_w = batch.bboxes[i].canvas_size
                    boxes = scale_boxes_to_letterbox(boxes, ori_h, ori_w, imgsz)
                    tgt_dict["boxes"] = boxes
                if batch.labels is not None and i < len(batch.labels):
                    tgt_dict["labels"] = batch.labels[i].to(device).long()
                if batch.masks is not None and i < len(batch.masks):
                    target_masks = batch.masks[i].data  # (N, ori_h, ori_w)
                    mask_ori_h, mask_ori_w = target_masks.shape[-2:]
                    scaled_masks = scale_masks_to_letterbox(target_masks, mask_ori_h, mask_ori_w, imgsz)
                    tgt_dict["masks"] = [encode_rle(m) for m in scaled_masks]
                target_list.append(tgt_dict)

            metric.update(preds=preds_list, target=target_list)

            now = time.perf_counter()
            iter_times.append(now - batch_start)
            batch_start = now

        results = metric.compute()
        formatted = self._format_torchmetrics_results(results)
        formatted.update(self._summarize_iter_times(iter_times))
        return formatted

    @staticmethod
    def _summarize_iter_times(iter_times: list[float]) -> dict[str, float]:
        """Average per-batch wall times into a single ``test/iter_time`` metric.

        Mirrors the trimmed-mean convention used elsewhere for iteration
        timing: the first batch is excluded since it includes one-off
        warmup costs (CUDA/XPU kernel compilation, first-call overhead)
        that would otherwise skew the average.
        """
        if not iter_times:
            return {}
        trimmed = iter_times[1:] if len(iter_times) > 1 else iter_times
        return {"test/iter_time": sum(trimmed) / len(trimmed)}

    def _test_classification_with_torchmetrics(
        self,
        metric_callable: Callable[[LabelInfo], Metric | MetricCollection],
        dataloader: DataLoader,
        yolo: YOLO,
        device: torch.device,
        imgsz: int | tuple[int, int],
    ) -> dict[str, float]:
        """Run torchmetrics evaluation for classification tasks.

        Supports both multi-class (softmax probabilities vs. class indices)
        and multi-label (sigmoid scores vs. multi-hot targets).
        """
        label_info = self._model.label_info or self._datamodule.label_info  # type: ignore[union-attr]
        metric = metric_callable(label_info)
        metric = metric.to(device)

        self._model.ensure_predict_ready()
        is_multilabel = getattr(self._model, "is_multilabel", False)

        for batch in dataloader:
            if not isinstance(batch, SampleBatch):
                msg = f"Expected test_dataloader to yield SampleBatch, got {type(batch)}"
                raise TypeError(msg)
            if batch.labels is None:
                msg = "Classification evaluation requires labels"
                raise TypeError(msg)

            imgs = batch.images.to(device) if isinstance(batch.images, torch.Tensor) else batch.images
            raw_results = yolo.predict(
                source=imgs,
                device=device,
                imgsz=imgsz,
                save=False,
                verbose=False,
            )

            scores = []
            for result in raw_results:
                if result.probs is None:  # pyrefly: ignore[missing-attribute]
                    msg = "Classification result is missing probabilities"
                    raise RuntimeError(msg)
                scores.append(torch.as_tensor(result.probs.data))
            preds = torch.stack(scores).to(device)
            targets = torch.stack([lbl.to(device) for lbl in batch.labels])
            targets = targets.float() if is_multilabel else targets.long().flatten()

            metric.update(preds=preds, target=targets)

        results = metric.compute()
        return self._format_torchmetrics_results(results)

    def _test_semantic_with_torchmetrics(
        self,
        metric_callable: Callable[[SegLabelInfo], Metric | MetricCollection],
        dataloader: DataLoader,
        yolo: YOLO,
        device: torch.device,
    ) -> dict[str, float]:
        """Run torchmetrics evaluation for semantic segmentation.

        Uses getitune's semantic segmentation metric (Dice + mIoU) on dense
        class maps produced by argmaxing the model logits.
        """
        label_info = self._model.label_info or self._datamodule.label_info  # type: ignore[union-attr]
        if not isinstance(label_info, SegLabelInfo):
            label_info = SegLabelInfo(
                label_names=label_info.label_names,
                label_groups=label_info.label_groups,
                label_ids=label_info.label_ids,
            )
        metric = metric_callable(label_info)
        metric = metric.to(device)

        if yolo.model is None:
            msg = "YOLO model is not loaded"
            raise RuntimeError(msg)
        yolo.model.to(device).eval()

        logger.info(
            f"Starting torchmetrics semantic segmentation evaluation: model={self._model.model_name}, "
            f"metric={type(metric).__name__}, batches={len(dataloader)}"
        )

        for batch in dataloader:
            if not isinstance(batch, SampleBatch):
                msg = f"Expected test_dataloader to yield SampleBatch, got {type(batch)}"
                raise TypeError(msg)
            if batch.masks is None:
                msg = "Semantic segmentation evaluation requires masks"
                raise TypeError(msg)

            imgs = batch.images.to(device) if isinstance(batch.images, torch.Tensor) else batch.images
            targets = torch.stack([torch.as_tensor(m.squeeze(0), dtype=torch.int32) for m in batch.masks]).to(device)

            with torch.no_grad():
                outputs = yolo.model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if outputs.shape[-2:] != targets.shape[-2:]:
                    outputs = f.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                preds = outputs.argmax(dim=1)

            metric.update(preds=preds, target=targets)

        results = metric.compute()
        return self._format_torchmetrics_results(results)

    def _compute_best_confidence_threshold(self) -> float | None:
        """Run FMeasure on the validation set to find the optimal confidence threshold.

        After training, the best threshold is stored in
        ``self._export_args["confidence_threshold"]`` so that export
        embeds the correct value into the model metadata.

        Returns:
            The best confidence threshold according to FMeasure, or ``None``
            if datamodule is unavailable.
        """
        if self._datamodule is None:
            return None

        # Load best checkpoint — after train() the model may hold last-epoch weights.
        if self._last_train_checkpoint is not None and self._last_train_checkpoint.exists():
            self._model.load_checkpoint(self._last_train_checkpoint)

        label_info = self._model.label_info or self._datamodule.label_info
        metric = FMeasure(label_info)
        device = self._device
        yolo = self._model.yolo
        metric = metric.to(device)

        self._datamodule.setup(stage="fit")
        dataloader = self._datamodule.val_dataloader()
        imgsz = self._model.imgsz

        logger.info("Computing best confidence threshold via FMeasure on validation set")

        for batch in dataloader:
            imgs = batch.images.to(device) if isinstance(batch.images, torch.Tensor) else batch.images
            raw_results = yolo.predict(
                source=imgs,
                device=device,
                imgsz=imgsz,
                conf=0.01,
                save=False,
                verbose=False,
            )

            preds_list = []
            for result in raw_results:
                pred_dict: dict[str, Any] = {
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                }
                if result.boxes is not None and len(result.boxes):
                    pred_dict["boxes"] = result.boxes.xyxy.to(device)  # pyrefly: ignore[missing-attribute]
                    pred_dict["scores"] = result.boxes.conf.to(device).float()  # pyrefly: ignore[missing-attribute]
                    pred_dict["labels"] = result.boxes.cls.to(device).long()  # pyrefly: ignore[missing-attribute]
                preds_list.append(pred_dict)

            target_list = []
            for i in range(len(raw_results)):
                tgt_dict: dict[str, Any] = {
                    "boxes": torch.zeros((0, 4), device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                }
                if batch.bboxes is not None and i < len(batch.bboxes):
                    boxes = batch.bboxes[i].data.to(device).float()
                    ori_h, ori_w = batch.bboxes[i].canvas_size
                    boxes = scale_boxes_to_letterbox(boxes, ori_h, ori_w, imgsz)
                    tgt_dict["boxes"] = boxes
                if batch.labels is not None and i < len(batch.labels):
                    tgt_dict["labels"] = batch.labels[i].to(device).long()
                target_list.append(tgt_dict)

            metric.update(preds=preds_list, target=target_list)

        metric.compute(best_confidence_threshold=None)
        return metric.best_confidence_threshold

    def _predict_with_datamodule(self, overrides: dict[str, Any]) -> list[Prediction]:
        """Run inference through ``DataModule.predict_dataloader()``."""
        if self._datamodule is None:
            msg = "_predict_with_datamodule requires a DataModule"
            raise TypeError(msg)
        overrides.pop("batch", None)
        dataloader = self._datamodule.predict_dataloader()

        yolo = self._model.yolo
        device = self._device
        self._model.ensure_predict_ready()

        predictions: list[Prediction] = []
        for batch in dataloader:
            if not isinstance(batch, SampleBatch):
                msg = f"Expected DataModule.predict_dataloader() to yield SampleBatch, got {type(batch)}"
                raise TypeError(msg)
            if not isinstance(batch.images, torch.Tensor):
                msg = f"Expected collated SampleBatch.images to be a tensor, got {type(batch.images)}"
                raise TypeError(msg)

            imgs = batch.images.to(device)
            raw_results = yolo.predict(
                source=imgs,
                device=device,
                imgsz=self._model.imgsz,
                save=False,
                verbose=False,
                **overrides,
            )
            predictions.extend(self._convert_predictions(raw_results, images=batch.images, imgs_info=batch.imgs_info))

        return predictions

    def _resolve_trainer_checkpoint(self, yolo: YOLO) -> Path | None:
        """Return the actual checkpoint produced by the latest training run."""
        trainer = getattr(yolo, "trainer", None)
        if trainer is None:
            return None

        for attr in ("best", "last"):
            checkpoint = getattr(trainer, attr, None)
            if checkpoint is None:
                continue
            checkpoint_path = Path(checkpoint).resolve()
            if checkpoint_path.exists():
                return checkpoint_path

        return None

    def _load_last_train_checkpoint(self) -> Path | None:
        """Load the persisted latest-training checkpoint pointer, if present."""
        checkpoint_file = self._work_dir / self._LAST_TRAIN_CHECKPOINT_FILE
        if not checkpoint_file.exists():
            return None

        checkpoint_text = checkpoint_file.read_text(encoding="utf-8").strip()
        if not checkpoint_text:
            return None

        checkpoint = Path(checkpoint_text)
        if checkpoint.exists():
            return checkpoint

        logger.warning(f"Ignoring stale checkpoint pointer: {checkpoint}")
        return None

    def _record_last_train_checkpoint(self, checkpoint: Path | None) -> None:
        """Persist the latest training checkpoint at a canonical location.

        Creates a copy at ``<work_dir>/best_checkpoint.pt`` to provide a
        backend-agnostic checkpoint path.
        """
        checkpoint_file = self._work_dir / self._LAST_TRAIN_CHECKPOINT_FILE
        if checkpoint is None:
            self._last_train_checkpoint = None
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            return

        # Copy to a canonical path so the public API never exposes
        # Ultralytics-internal directory structure (train/weights/best.pt).
        # Unlink any pre-existing symlink so copyfile writes a real file
        # instead of following the symlink and overwriting its target.
        canonical_path = self._work_dir / "best_checkpoint.pt"
        if canonical_path.is_symlink():
            canonical_path.unlink()
        shutil.copyfile(checkpoint, canonical_path)

        self._last_train_checkpoint = canonical_path
        checkpoint_file.write_text(str(canonical_path), encoding="utf-8")

    def _remap_results_csv(self) -> None:
        """Rename columns in the Ultralytics ``results.csv`` to standard metric names.

        Ultralytics writes its native metric names (e.g. ``train/box_loss``,
        ``metrics/mAP50(B)``) as CSV column headers.  This method renames them
        in-place using the model's ``metric_keys`` mapping so that downstream
        consumers (like the application backend) receive backend-agnostic names.
        """
        results_csv = self._work_dir / "train" / "results.csv"
        if not results_csv.exists():
            return

        with results_csv.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return

        # Remap header row using the model's metric_keys.
        header = rows[0]
        mapping = self._model.metric_keys
        new_header = [mapping.get(col.strip(), col.strip()) for col in header]

        # Populate synthetic ``step`` column from ``epoch`` when absent.
        # Ultralytics logs everything per-epoch, but the Geti application parser
        # expects a ``step`` column for ``frequency="step"`` metrics.
        epoch_idx = next((i for i, h in enumerate(new_header) if h.strip() == "epoch"), None)
        step_idx = next((i for i, h in enumerate(new_header) if h.strip() == "step"), None)

        if step_idx is None and epoch_idx is not None:
            # No step column — insert one after epoch.
            new_header.insert(epoch_idx + 1, "step")
            for row in rows[1:]:
                epoch_val = row[epoch_idx] if epoch_idx < len(row) else ""
                row.insert(epoch_idx + 1, epoch_val)

        rows[0] = new_header

        with results_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        # Write the same remapped content to csv/version_0/metrics.csv to match
        # Lightning's CSVLogger output structure, so downstream consumers can
        # always find metrics at a single known path.
        csv_version_dir = self._work_dir / "csv" / "version_0"
        csv_version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(results_csv, csv_version_dir / "metrics.csv")
        logger.info(f"Remapped results.csv columns to standard metric names: {results_csv}")

    def _make_bound_trainer(
        self,
        progress_fn: Callable[[float], None] | None = None,
        progress_min: float = 0.0,
        progress_max: float = 100.0,
        max_grad_norm: float | None = None,
    ) -> type:
        """Return a trainer subclass with the DataModule bound as a class attr."""
        base_cls = self._model.trainer_cls
        if base_cls is None:
            msg = f"{type(self._model).__name__} does not define a trainer_cls"
            raise TypeError(msg)

        if self._datamodule is None and max_grad_norm is None:
            return base_cls

        attrs: dict[str, Any] = {
            "_datamodule": self._datamodule,
            "_use_getitune_data": self._datamodule is not None,
            "_progress_fn": progress_fn,
            "_progress_min": progress_min,
            "_progress_max": progress_max,
        }
        if max_grad_norm is not None:
            attrs["_max_grad_norm"] = max_grad_norm
        return type(base_cls.__name__, (base_cls,), attrs)

    @staticmethod
    def _extract_progress_callback(
        callbacks: list[Any] | None,
    ) -> tuple[Callable[[float], None] | None, float, float]:
        """Extract progress reporting callable from Lightning-style callbacks.

        Scans for any callback with ``_on_progress_update``, ``_min_p``, and
        ``_max_p`` attributes (duck-typed to avoid coupling to the application
        backend's ``TrainingProgressCallback``).

        Returns:
            (progress_fn, min_p, max_p) or (None, 0, 100) if not found.
        """
        if callbacks is None:
            return None, 0.0, 100.0

        for cb in callbacks:
            fn = getattr(cb, "_on_progress_update", None)
            if fn is not None:
                min_p = getattr(cb, "_min_p", 0.0)
                max_p = getattr(cb, "_max_p", 100.0)
                return fn, min_p, max_p

        return None, 0.0, 100.0

    def _make_bound_validator(self) -> type:
        """Return a validator subclass with the DataModule bound as a class attr."""
        base_cls = self._model.validator_cls
        if base_cls is None:
            msg = f"{type(self._model).__name__} does not define a validator_cls"
            raise TypeError(msg)

        if self._datamodule is None:
            return base_cls

        return type(base_cls.__name__, (base_cls,), {"_datamodule": self._datamodule, "_use_getitune_data": True})

    def _build_overrides(self, defaults: Mapping[str, Any] | None = None, **kwargs) -> dict[str, Any]:
        """Merge overrides: model defaults < engine kwargs < defaults < call kwargs."""
        overrides: dict[str, Any] = {}
        overrides.update(self._model.extra_overrides)
        overrides.update(self._kwargs)
        if defaults is not None:
            overrides.update(defaults)
        overrides.update(kwargs)
        return overrides

    def _translate_metrics(self, results: UMETRICS | dict[str, Any] | None) -> dict[str, float]:
        """Map Ultralytics metric keys to getitune names.

        Translates aggregate metrics via ``metric_keys`` and extracts
        per-class precision, recall, mAP50, and mAP50-95 when available.
        Per-class keys use the format ``val/<metric>/<class_name>``.
        """
        if results is None:
            return {}

        raw_metrics: dict[str, float] = {}
        if hasattr(results, "results_dict"):
            raw_metrics = dict(results.results_dict)
        elif isinstance(results, dict):
            raw_metrics = dict(results)

        translated: dict[str, float] = {}
        for ultra_key, value in raw_metrics.items():
            gt_key = self._model.metric_keys.get(ultra_key, f"ultralytics/{ultra_key}")
            translated[gt_key] = float(value) if not isinstance(value, float) else value

        self._add_per_class_metrics(results, translated)
        self._add_speed_metric(results, translated)
        return translated

    @staticmethod
    def _add_speed_metric(results: UMETRICS | dict[str, Any], metrics: dict[str, float]) -> None:
        """Translate Ultralytics' per-image ``speed`` dict into ``val/iter_time``.

        The Ultralytics validator (``BaseValidator``) tracks per-image
        ``preprocess``/``inference``/``loss``/``postprocess`` timings (in
        milliseconds) on a ``speed`` dict that is copied onto the returned
        metrics object. Summing and converting to seconds gives a
        cross-backend-comparable per-iteration time, mirroring Lightning's
        ``IterationTimer`` (``val/iter_time`` / ``test/iter_time``).
        """
        speed = getattr(results, "speed", None)
        if not isinstance(speed, dict) or not speed:
            return
        total_ms = sum(float(v) for v in speed.values())
        metrics["val/iter_time"] = total_ms / 1000.0

    @staticmethod
    def _add_per_class_metrics(results: UMETRICS | dict[str, Any], metrics: dict[str, float]) -> None:
        """Extract per-class metrics from the Ultralytics results object.

        Adds keys like ``val/precision/<ClassName>``, ``val/recall/<ClassName>``,
        ``val/map_50/<ClassName>``, ``val/map/<ClassName>`` to *metrics* in-place.
        """
        names = getattr(results, "names", None)
        ap_class_index = getattr(results, "ap_class_index", None)
        if names is None or ap_class_index is None:
            return

        for i, cls_idx in enumerate(ap_class_index):
            class_name = names.get(cls_idx, str(cls_idx))
            try:
                result = results.class_result(i)  # pyrefly: ignore[missing-attribute]
            except (IndexError, AttributeError, TypeError):
                continue
            # Detection returns 4 values (p, r, ap50, ap)
            # Segmentation returns 8 values (box p, r, ap50, ap, mask p, r, ap50, ap)
            if len(result) >= 4:
                p, r, ap50, ap = result[:4]
                metrics[f"val/precision/{class_name}"] = float(p)
                metrics[f"val/recall/{class_name}"] = float(r)
                metrics[f"val/map_50/{class_name}"] = float(ap50)
                metrics[f"val/map/{class_name}"] = float(ap)
            if len(result) >= 8:
                mp, mr, map50, map_ = result[4:8]
                metrics[f"val/mask_precision/{class_name}"] = float(mp)
                metrics[f"val/mask_recall/{class_name}"] = float(mr)
                metrics[f"val/mask_map_50/{class_name}"] = float(map50)
                metrics[f"val/mask_map/{class_name}"] = float(map_)

    @staticmethod
    def _format_torchmetrics_results(results: dict[str, Any]) -> dict[str, float]:
        """Convert torchmetrics compute output to a flat ``test/``-prefixed dict.

        Mirrors the logic in ``LightningModel._log_metrics``: only scalar
        tensors are included; auxiliary keys (``classes``, ``map_per_class``,
        ``mar_100_per_class``, ``ious``) are skipped. Nested dicts returned by
        ``MetricCollection`` are flattened recursively.

        Args:
            results: Dict returned by ``metric.compute()``.

        Returns:
            Flat dict, e.g. ``{"test/map": 0.75, "test/map_50": 0.90}``.
        """
        _skip_keys = {"classes", "map_per_class", "mar_100_per_class", "ious"}
        formatted: dict[str, float] = {}

        def _add(prefix: str, value: Any) -> None:  # noqa: ANN401
            if isinstance(value, dict):
                for key, nested in value.items():
                    if prefix.endswith(f"/{key}"):
                        _add(prefix, nested)
                    else:
                        _add(f"{prefix}/{key}", nested)
                return
            if prefix.rsplit("/", 1)[-1] in _skip_keys:
                return
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    formatted[prefix] = value.item()
                else:
                    logger.debug(f"Skipping non-scalar torchmetric '{prefix}' with {value.numel()} elements")
            elif isinstance(value, (int, float)):
                formatted[prefix] = float(value)

        for name, value in results.items():
            _add(f"test/{name}", value)
        return formatted

    def _convert_predictions(
        self,
        raw_results: list[Any],
        images: torch.Tensor | tv_tensors.Image | list[torch.Tensor] | list[tv_tensors.Image] | None = None,
        imgs_info: Sequence[ImageInfo | None] | None = None,
    ) -> list[Prediction]:
        """Convert Ultralytics ``Results`` to getitune ``Prediction``.

        Handles three output shapes:

        * **Detection** (``result.boxes``): bounding boxes + class scores.
        * **Instance segmentation** (``result.boxes`` + ``result.masks``):
          same as detection, plus a ``(N, H, W)`` mask tensor.
        * **Classification** (``result.probs``): per-class probability vector.
          Multi-class returns the top-1 label; multi-label returns all labels
          above the default 0.5 threshold. ``bboxes`` and ``masks`` remain
          ``None``.
        * **Semantic segmentation** (``result.semantic_mask``): dense per-pixel
          class map returned as a single-channel ``tv_tensors.Mask``.
        """
        predictions: list[Prediction] = []
        for idx, result in enumerate(raw_results):
            img_tensor, img_info = UltralyticsEngine._resolve_prediction_input(idx, result, images, imgs_info)
            h, w = img_info.ori_shape

            bboxes = None
            scores = None
            labels = None
            masks = None

            # --- classification ---
            probs = getattr(result, "probs", None)
            if probs is not None:
                scores = torch.as_tensor(probs.data).cpu().float()  # (nc,)
                if getattr(self._model, "is_multilabel", False):
                    labels = (scores >= 0.5).nonzero(as_tuple=False).flatten().long()
                else:
                    labels = torch.tensor([int(probs.top1)], dtype=torch.long)

            # --- semantic segmentation ---
            semantic_mask = getattr(result, "semantic_mask", None)
            if semantic_mask is not None:
                semantic_mask_data = getattr(semantic_mask, "data", semantic_mask)
                semantic_tensor = torch.as_tensor(semantic_mask_data).cpu()
                if semantic_tensor.ndim == 2:
                    semantic_tensor = semantic_tensor.unsqueeze(0)
                masks = tv_tensors.Mask(semantic_tensor)

            # --- detection / instance-segmentation ---
            elif result.boxes is not None and len(result.boxes):
                boxes_xyxy = torch.as_tensor(result.boxes.xyxy).cpu()  # pyrefly: ignore[missing-attribute]
                bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    boxes_xyxy,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(h, w),
                )
                scores = torch.as_tensor(result.boxes.conf).cpu()  # pyrefly: ignore[missing-attribute]
                labels = torch.as_tensor(result.boxes.cls).cpu().long()  # pyrefly: ignore[missing-attribute]

                if result.masks is not None and len(result.masks):
                    masks = tv_tensors.Mask(
                        torch.as_tensor(result.masks.data).cpu()
                    )  # pyrefly: ignore[missing-attribute]

            predictions.append(
                Prediction(
                    image=tv_tensors.Image(img_tensor),
                    img_info=img_info,
                    bboxes=bboxes,
                    scores=scores,
                    label=labels,
                    masks=masks,
                ),
            )

        return predictions

    @staticmethod
    def _resolve_prediction_input(
        idx: int,
        result: _UltralyticsResultLike,
        images: torch.Tensor | tv_tensors.Image | list[torch.Tensor] | list[tv_tensors.Image] | None,
        imgs_info: Sequence[ImageInfo | None] | None,
    ) -> tuple[torch.Tensor, ImageInfo]:
        """Use batch inputs when available, otherwise fall back to Ultralytics results."""
        if images is not None:
            image = images[idx]
            img_tensor = torch.as_tensor(image).detach().cpu().float()
            img_info = imgs_info[idx] if imgs_info is not None else None
            if img_info is None:
                _, h, w = img_tensor.shape
                img_info = ImageInfo(  # pyrefly: ignore[no-matching-overload]
                    img_idx=idx,
                    img_shape=(h, w),
                    ori_shape=(h, w),
                )
            return img_tensor, img_info

        img_tensor = torch.from_numpy(result.orig_img).permute(2, 0, 1).float()
        h, w = result.orig_shape[0], result.orig_shape[1]
        img_info = ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=idx,
            img_shape=(h, w),
            ori_shape=(h, w),
        )
        return img_tensor, img_info

    @staticmethod
    def _resolve_device(device: str | DeviceType) -> torch.device:
        """Resolve a device specification to a :class:`torch.device`.

        Resolution order for ``"auto"`` / ``DeviceType.auto``:
        XPU > CUDA > CPU (matches getitune convention).

        We return a ``torch.device`` object rather than a plain string so that
        Ultralytics' ``select_device()`` passes it through unchanged — its
        validator only rejects *string* device names it doesn't recognise,
        but accepts pre-constructed ``torch.device`` objects verbatim.

        Args:
            device: Raw string (``"auto"``, ``"xpu"``, ``"xpu:0"``, ``"cuda"``,
                ``"cuda:0"``, ``"0"``, ``"cpu"``) or :class:`DeviceType` enum.

        Returns:
            Resolved ``torch.device``.
        """
        # Normalise DeviceType enum to string.
        if isinstance(device, DeviceType):
            device = {
                DeviceType.auto: "auto",
                DeviceType.xpu: "xpu",
                DeviceType.gpu: "cuda",
                DeviceType.cpu: "cpu",
            }.get(device, str(device.value))

        device = str(device).strip().lower()

        if device == "auto":
            if is_xpu_available():
                return torch.device("xpu")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        if device == "xpu":
            return torch.device("xpu")

        if device in ("cuda", "gpu"):
            return torch.device("cuda")

        # Bare integer index → CUDA device (Ultralytics convention).
        if device.isdigit():
            return torch.device(f"cuda:{device}")

        # Anything else (e.g. "cuda:1", "cpu") — let torch.device parse it.
        return torch.device(device)

    @staticmethod
    def _precision_to_amp(precision: _PRECISION_INPUT | None, device: torch.device) -> bool | None:
        """Map a Lightning-style precision string to Ultralytics' ``amp`` flag.

        Args:
            precision: One of Lightning supported precisions: ``64, 32, 16,
                'transformer-engine', 'transformer-engine-float16',
                '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true',
                '64-true', '64', '32', '16', 'bf16'``,
                or ``None`` to leave the Ultralytics default (``amp=True``).
            device: The training device.  CPU does not support AMP; any FP16/
                BF16 variant is downgraded to FP32 with a warning.

        Returns:
            ``True`` to enable AMP, ``False`` to disable it, or ``None`` to
            leave the Ultralytics default unchanged.
        """
        if precision is None:
            return None

        if precision in (
            "16-mixed",
            "16",
            16,
            "bf16-mixed",
            "bf16",
            "16-true",
            "16-mix",
            "transformer-engine-float16",
            "bf16-true",
        ):
            if device.type == "cpu":
                logger.warning(f"precision={precision!r} is not supported on CPU; falling back to FP32 (amp=False)")
                return False
            return True

        return False
