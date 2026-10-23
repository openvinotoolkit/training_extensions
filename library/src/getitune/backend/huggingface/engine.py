# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face engine implementation."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import accelerate
import torch
import transformers
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments

from getitune.data.entity.sample import Prediction
from getitune.data.module import DataModule
from getitune.engine.engine import Engine
from getitune.types.device import DeviceType
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision
from getitune.types.task import TaskType
from getitune.utils.device import is_xpu_available

from .callbacks.epoch_summary import HFEpochSummaryCallback
from .callbacks.progress import HFProgressCallback, extract_progress_fn
from .engine_utils import resolve_greater_is_better, resolve_precision, summarize_log_history, unbatch_predictions
from .models.base import HFModel
from .plugins.xpu import XPUMemoryCallback
from .tools.configurator import Configurator
from .trainers import GetiTuneHFTrainer, write_metrics_csv

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lightning_fabric.plugins.precision.precision import _PRECISION_INPUT

    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.types import ANNOTATIONS, DATA, METRICS, MODEL

logger = logging.getLogger(__name__)
logger.debug("Hugging Face backend: transformers=%s accelerate=%s", transformers.__version__, accelerate.__version__)

# Fallback monitor key used when a recipe/training config does not specify one.
_DEFAULT_MONITOR: dict[TaskType, str] = {
    TaskType.DETECTION: "val/map",
    TaskType.INSTANCE_SEGMENTATION: "val/map",
    TaskType.MULTI_LABEL_CLS: "val/map",
    TaskType.MULTI_CLASS_CLS: "val/f1-score",
    TaskType.SEMANTIC_SEGMENTATION: "val/Dice",
}


class HFEngine(Engine):
    """Engine backed by ``transformers``.

    Wraps an :class:`HFModel` and a :class:`~getitune.data.module.DataModule`
    (or data-root path). Selected by ``backend: huggingface`` in a recipe.
    """

    _EXPORTED_MODEL_BASE_NAME: ClassVar[str] = "exported_model"
    _CHECKPOINT_DIR_NAME: ClassVar[str] = "best_checkpoint"
    backend_name: ClassVar[str] = "huggingface"
    model_base_class: ClassVar[type] = HFModel

    def __init__(
        self,
        model: HFModel,
        data: DataModule | PathLike,
        work_dir: PathLike | None = None,
        device: str | DeviceType = "auto",
        checkpoint: PathLike | None = None,
        training: dict[str, Any] | None = None,
        num_devices: int = 1,
        **kwargs,
    ) -> None:
        """Initialize the engine.

        Args:
            model: Hugging Face model wrapper.
            data: DataModule or filesystem data-root path.
            work_dir: Directory for checkpoints, exports and logs.
                Defaults to ``"./getitune-workspace"``.
            device: Device string or :class:`DeviceType`
                (``"auto"``, ``"xpu"``, ``"cpu"``, ``"gpu"``).
            checkpoint: Optional ``save_pretrained()`` directory to load
                weights from before training.
            training: Default training hyperparameters (``max_epochs``,
                ``batch``, ``learning_rate``, ``patience``, ``precision``),
                normally supplied by :class:`~getitune.backend.huggingface.tools.configurator.Configurator`
                from a recipe's ``training:`` block. Only used to fill in
                whichever of :meth:`train`'s parameters are left as ``None``
                — explicit call-site arguments always win.
            num_devices: Number of distributed processes to use. Values above
                one require launching the training script with ``torchrun``.
            **kwargs: Extra keyword arguments accepted for parity with other
                engines' constructors (e.g. ``task=`` forwarded by
                ``create_engine``); unused by this backend.

        Raises:
            TypeError: If *model* is not an :class:`HFModel`, or *data* is
                neither a :class:`DataModule` nor a path.
        """
        if not isinstance(model, HFModel):
            msg = f"model must be an HFModel instance, got {type(model)}"
            raise TypeError(msg)
        if num_devices < 1:
            msg = f"num_devices must be at least 1, got {num_devices}"
            raise ValueError(msg)

        self._model = model
        self._work_dir = Path(work_dir or "./getitune-workspace").resolve()
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._device = self._resolve_accelerator(device)
        self._training_defaults = dict(training or {})
        self.num_devices = num_devices

        if isinstance(data, DataModule):
            self._datamodule: DataModule | None = data
            self._data_root: Path | None = None
        elif isinstance(data, (str, os.PathLike)):
            self._datamodule = None
            self._data_root = Path(data)
        else:
            msg = f"data must be DataModule or PathLike, got {type(data)}"
            raise TypeError(msg)

        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)

        # The exporter reads preprocessing metadata off the model, so the
        # DataModule's intensity settings have to reach it before export.
        if self._datamodule is not None:
            intensity_cfg = getattr(self._datamodule, "input_intensity_config", None)
            if intensity_cfg is not None:
                self._model.set_intensity_config(intensity_cfg)

    def train(
        self,
        max_epochs: int | None = None,
        batch: int | None = None,
        learning_rate: float | None = None,
        patience: int | None = None,
        precision: _PRECISION_INPUT | None = None,
        callbacks: list | None = None,
        checkpoint: PathLike | None = None,
        metric: MetricCallable | None = None,
        val_check_interval: int | None = None,
        monitor: str | None = None,
        devices: int | list[int] | None = None,
        **kwargs,
    ) -> METRICS:
        """Train the model.

        Args:
            max_epochs: Maximum number of epochs. Falls back to the recipe's
                ``training.max_epochs`` (see *training* in
                :meth:`HFEngine.__init__`), then to 100.
            batch: Batch size override. Falls back to the recipe's
                ``training.batch``, then to each subset's configured
                ``batch_size``.
            learning_rate: Initial learning rate. Falls back to the recipe's
                ``training.learning_rate``, then to 5e-5.
            patience: Early-stopping patience, in evaluation rounds. Falls
                back to the recipe's ``training.patience``. Only takes effect
                when a non-empty validation split is attached.
            precision: Same value space as Lightning's ``_PRECISION_INPUT``
                (``16``, ``32``, ``"bf16-mixed"``, etc.). Falls back to the
                recipe's ``training.precision``, then to ``"bf16-mixed"``
                since XPU is the primary accelerator.
            callbacks: Training callbacks. Objects with
                ``_on_progress_update``/``_min_p``/``_max_p`` are consumed for
                progress reporting (G18); genuine
                ``transformers.TrainerCallback`` instances are passed through
                to ``Trainer`` unchanged; anything else is dropped.
            checkpoint: Optional checkpoint to resume or warm-start from.
            metric: Optional metric callable taking ``label_info`` and
                returning a ``torchmetrics.Metric``/``MetricCollection``. If
                ``None``, the model's default task metric is used.
            val_check_interval: Run validation and the best-checkpoint check
                every N epochs. ``1`` means every epoch (default). Falls back
                to the recipe's ``training.val_check_interval``.
            monitor: Metric key used for best checkpointing and early
                stopping, e.g. ``"val/map"`` or ``"val/Dice"``. Falls back to
                the recipe's ``training.monitor``, then to a task default.
            devices: device selection. A one-element
                list selects that CUDA/XPU device for a single-process run;
                an integer or longer list specifies the process count for
                torchrun, which assigns devices from the launcher's visible
                devices by ``LOCAL_RANK``.
            **kwargs: Additional ``transformers.TrainingArguments`` overrides.

        Returns:
            The best validation metrics merged with the final training
            summary, keyed with the same ``train/`` / ``val/`` convention as
            the metrics CSV.

        Raises:
            ValueError: If the engine was built from a bare data-root path
                rather than a :class:`DataModule`.
        """
        if self._datamodule is None:
            msg = (
                "HFEngine.train() requires a DataModule to build dataloaders from; "
                "got a bare data-root path. Build a DataModule first, e.g. via "
                "DataModule(...) directly, or by resolving a recipe through "
                "HFEngine.from_config() / create_engine()."
            )
            raise ValueError(msg)

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        requested_num_devices, selected_device_index = self._resolve_devices(
            devices, self._device.type, self.num_devices
        )
        if world_size != requested_num_devices:
            msg = (
                f"HFEngine is configured for {requested_num_devices} device(s), but WORLD_SIZE={world_size}. "
                "For multi-device training, launch with torchrun and set devices to the process count."
            )
            raise ValueError(msg)
        self.num_devices = requested_num_devices
        if selected_device_index is not None:
            self._device = torch.device(self._device.type, selected_device_index)

        defaults = self._training_defaults
        max_epochs = max_epochs if max_epochs is not None else defaults.get("max_epochs")
        batch = batch if batch is not None else defaults.get("batch")
        learning_rate = learning_rate if learning_rate is not None else defaults.get("learning_rate")
        patience = patience if patience is not None else defaults.get("patience")
        precision = precision if precision is not None else defaults.get("precision", "bf16-mixed")
        val_check_interval = (
            val_check_interval if val_check_interval is not None else defaults.get("val_check_interval")
        )
        monitor = monitor if monitor is not None else defaults.get("monitor")

        # The named keys above are consumed here and excluded to
        # avoid double-setting them.
        named_keys = {
            "max_epochs",
            "batch",
            "learning_rate",
            "patience",
            "precision",
            "val_check_interval",
            "monitor",
        }
        extra_training_args = {k: v for k, v in defaults.items() if k not in named_keys}

        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)

        fp16, bf16 = resolve_precision(precision)
        progress_fn, min_p, max_p = extract_progress_fn(callbacks)
        if self._device.index is not None:
            if world_size > 1:
                msg = (
                    "A single-device selection is for single-process runs. For torchrun, select devices through the "
                    "launcher-visible device configuration; each process is assigned by LOCAL_RANK."
                )
                raise ValueError(msg)
            getattr(torch, self._device.type).set_device(self._device.index)

        trainer_callbacks: list[Any] = [cb for cb in (callbacks or []) if isinstance(cb, TrainerCallback)]
        if progress_fn is not None:
            trainer_callbacks.append(HFProgressCallback(progress_fn, min_p, max_p))
        if not any(isinstance(cb, HFEpochSummaryCallback) for cb in trainer_callbacks):
            trainer_callbacks.append(HFEpochSummaryCallback())
        if is_xpu_available():
            trainer_callbacks.append(XPUMemoryCallback())

        val_subset = self._datamodule.subsets.get("val")
        has_eval = bool(val_subset) and len(val_subset) > 0
        monitor = monitor if monitor is not None else _DEFAULT_MONITOR.get(self._model.task)

        training_args_kwargs: dict[str, Any] = {
            "output_dir": str(self._work_dir / "train"),
            "num_train_epochs": max_epochs if max_epochs is not None else 100,
            "per_device_train_batch_size": batch or self._datamodule.train_subset.batch_size,
            "per_device_eval_batch_size": batch or self._datamodule.val_subset.batch_size,
            "learning_rate": learning_rate if learning_rate is not None else 5e-5,
            "fp16": fp16,
            "bf16": bf16,
            "remove_unused_columns": False,
            "report_to": [],
            "label_names": list(self._model.label_keys),
            "use_cpu": self._device.type == "cpu",
            "eval_strategy": "epoch" if has_eval else "no",
            "save_strategy": "no",
            "logging_strategy": "steps",
            "logging_steps": 1,
        }
        if has_eval and monitor is not None:
            greater_is_better = resolve_greater_is_better(monitor)
            training_args_kwargs.update(
                load_best_model_at_end=False,
                metric_for_best_model=monitor,
                greater_is_better=greater_is_better,
            )
            if patience is not None:
                trainer_callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        training_args_kwargs.update(extra_training_args)
        training_args_kwargs.update(kwargs)

        # ReduceLROnPlateau steps the metric from ``metric_for_best_model``, but torch's
        # default mode is "min" (watched metric must decrease). Our targets (mAP, F1, Dice,
        # accuracy) are maximized, so inject the mode matching ``greater_is_better`` unless
        # the caller already set it explicitly.
        scheduler_kwargs = training_args_kwargs.get("lr_scheduler_kwargs")
        scheduler_type = training_args_kwargs.get("lr_scheduler_type", "linear")
        scheduler_type = getattr(scheduler_type, "value", scheduler_type)
        if scheduler_type == "reduce_lr_on_plateau":
            scheduler_kwargs = dict(scheduler_kwargs) if isinstance(scheduler_kwargs, dict) else {}
            if "mode" not in scheduler_kwargs:
                greater = training_args_kwargs.get(
                    "greater_is_better", resolve_greater_is_better(training_args_kwargs.get("metric_for_best_model"))
                )
                scheduler_kwargs["mode"] = "max" if greater else "min"
            training_args_kwargs["lr_scheduler_kwargs"] = scheduler_kwargs

        # Convert`warmup_ratio` to the supported `warmup_steps` using the estimated number of training steps.
        # Warmup steps count optimizer updates, so the loader batch count must be divided by
        # gradient_accumulation_steps (math.ceil matches transformers' own optimizer-step estimation).
        warmup_ratio = training_args_kwargs.pop("warmup_ratio", None)
        if warmup_ratio is not None:
            train_batch = training_args_kwargs["per_device_train_batch_size"]
            accumulation_steps = training_args_kwargs.get("gradient_accumulation_steps", 1) or 1
            steps_per_epoch = math.ceil(len(self._datamodule.subsets["train"]) / (train_batch * accumulation_steps))
            total_steps = max(1, int(steps_per_epoch * training_args_kwargs["num_train_epochs"]))
            training_args_kwargs["warmup_steps"] = int(warmup_ratio * total_steps)

        args = TrainingArguments(**training_args_kwargs)
        if world_size == 1 and args.n_gpu > 1:
            # Per-image annotations cannot be scattered safely by DataParallel.
            args._n_gpu = 1  # noqa: SLF001
        if self._device.index is not None:
            # Transformers resets single-process runs to cuda:0 during setup.
            args._setup_devices = self._device  # noqa: SLF001
            getattr(torch, self._device.type).set_device(self._device)
        val_subset = self._datamodule.subsets.get("val")
        trainer = GetiTuneHFTrainer(
            self._model,
            self._datamodule,
            metric=metric,
            val_check_interval=val_check_interval,
            model=self._model.hf_model,
            args=args,
            train_dataset=self._datamodule.subsets["train"],
            eval_dataset=val_subset if has_eval else None,
            callbacks=trainer_callbacks,
        )
        trainer.train()
        trainer.accelerator.wait_for_everyone()

        best_dir = self._work_dir / self._CHECKPOINT_DIR_NAME
        if best_dir.exists():
            self._model.load_checkpoint(best_dir)
        else:
            trainer.save_model(str(best_dir))
        trainer.accelerator.wait_for_everyone()
        self._model.record_checkpoint(best_dir)
        # Extract only when the selected checkpoint carries no threshold: the
        # best-checkpoint save persists its *same-epoch* validation threshold
        # (see ``_maybe_save_best_checkpoint``), so a restore above already
        # has the authoritative value. Extraction here is the fallback for
        # runs without such a save (e.g. no monitored metric).
        if self._model.best_confidence_threshold is None:
            self._extract_best_confidence_threshold(trainer)

        if trainer.is_world_process_zero():
            write_metrics_csv(trainer.state.log_history, self._work_dir)

        summary = summarize_log_history(trainer.state.log_history)
        summary.update(trainer._best_eval_metrics)  # noqa: SLF001
        return summary

    def test(
        self,
        metric: MetricCallable | None = None,
        batch: int | None = None,
        checkpoint: PathLike | None = None,
        **kwargs,
    ) -> METRICS:
        """Evaluate the model on the test split.

        Bypasses ``transformers.Trainer.evaluate()``: see
        ``GetiTuneHFTrainer.evaluate(split="test")`` for why. Results are
        computed by the model's own :meth:`HFModel.build_default_metric`
        (or an override), then flattened to ``test/`` scalar keys, matching
        the naming convention ``LightningEngine.test()`` uses.

        Args:
            metric: A callable taking ``label_info`` and returning a
                ``torchmetrics`` ``Metric``/``MetricCollection``, used
                instead of the model's default metric.
            batch: Batch size override. Defaults to the test subset's
                configured ``batch_size``.
            checkpoint: Optional checkpoint to load before evaluating. If
                ``None``, ``self.best_checkpoint`` is used, matching
                :meth:`export` and ``LightningEngine.test()``.
            **kwargs: Additional ``transformers.TrainingArguments`` overrides.

        Returns:
            Scalar metrics keyed with the ``test/`` prefix.

        Raises:
            ValueError: If the engine was built from a bare data-root path,
                or the attached DataModule has no (non-empty) test split.
        """
        if self._datamodule is None:
            msg = "HFEngine.test() requires a DataModule to build a dataloader from; got a bare data-root path."
            raise ValueError(msg)

        if checkpoint is None:
            checkpoint = self.best_checkpoint
        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)

        trainer = self._build_test_scoped_trainer("test", batch, kwargs)

        metric_obj = metric(self._model.label_info) if metric is not None else self._model.build_default_metric()
        compute_kwargs = (
            {"best_confidence_threshold": self._model.best_confidence_threshold}
            if metric is None and self._model.best_confidence_threshold is not None
            else None
        )
        return trainer.evaluate(split="test", metric=metric_obj, compute_kwargs=compute_kwargs)

    def _extract_best_confidence_threshold(self, trainer: GetiTuneHFTrainer) -> None:
        """Store the validation F1-optimal confidence threshold on the model.

        The detection / instance-segmentation default metric collection contains
        an ``FMeasure`` submetric whose threshold sweep computes the best
        confidence threshold on the validation set each epoch. After training
        it is persisted on the model so export and ``predict()`` pick it up.
        """
        from getitune.metrics.fmeasure import FMeasure

        metric_obj = trainer._val_metric  # noqa: SLF001
        if metric_obj is None:
            return
        fmeasure = getattr(metric_obj, "FMeasure", None)
        if fmeasure is None and isinstance(metric_obj, FMeasure):
            fmeasure = metric_obj
        if fmeasure is None:
            return
        # Read the backing attribute directly: the ``best_confidence_threshold``
        # property raises ``RuntimeError`` instead of returning ``None`` while
        # the metric has never computed (e.g. training without a validation
        # metric), which would abort an otherwise completed run.
        threshold = fmeasure._current_confidence_threshold  # noqa: SLF001
        if threshold is None:
            return
        self._model.best_confidence_threshold = float(threshold)
        logger.info("Best confidence threshold from validation F-measure: %s", float(threshold))

    def predict(
        self,
        confidence_threshold: float | None = None,
        batch: int | None = None,
        checkpoint: PathLike | None = None,
        **kwargs,
    ) -> ANNOTATIONS:
        """Run inference on the test split.

        Like ``test()``, this always runs over the attached DataModule's
        test split rather than an arbitrary source path — the same
        convention ``LightningEngine.predict()`` and
        ``DataModule.predict_dataloader()`` use (predicting is "run over the
        test images", ground truth simply unused). A bare data-root path
        (no attached source-image loader) is not supported.

        Args:
            confidence_threshold: Minimum score for a detection / instance
                segmentation prediction to be kept. Defaults to the best
                confidence threshold computed by validation F-measure (or the
                model's configured default when not available). Not applied to classification
                (whose scores are per-class probabilities, not a filtering
                signal) or semantic segmentation (which has no per-prediction
                score at all).
            batch: Batch size override. Defaults to the test subset's
                configured ``batch_size``.
            checkpoint: Optional checkpoint to load before predicting.
            **kwargs: Additional ``transformers.TrainingArguments`` overrides.

        Returns:
            One prediction per test-split image.

        Raises:
            ValueError: If the engine was built from a bare data-root path,
                or the attached DataModule has no (non-empty) test split.
        """
        if self._datamodule is None:
            msg = "HFEngine.predict() requires a DataModule to build a dataloader from; got a bare data-root path."
            raise ValueError(msg)

        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)

        if confidence_threshold is None:
            confidence_threshold = (
                self._model.best_confidence_threshold
                if self._model.best_confidence_threshold is not None
                else self._model.default_confidence_threshold
            )

        trainer = self._build_test_scoped_trainer("predict", batch, kwargs)

        predictions: list[Prediction] = []
        for pred_batch in trainer.predict_batches():
            predictions.extend(unbatch_predictions(pred_batch, confidence_threshold))
        return predictions  # pyrefly: ignore[bad-return]

    def _build_test_scoped_trainer(self, subdir: str, batch: int | None, kwargs: dict[str, Any]) -> GetiTuneHFTrainer:
        """Build a :class:`GetiTuneHFTrainer` scoped to the test split, shared by ``test()``/``predict()``.

        Both callers need the identical setup (test-split guard, output dir,
        eval batch size, device flag), so it lives here once rather than
        twice. No ``train_dataset`` is passed — ``Trainer.__init__`` doesn't
        validate its presence at construction time, only ``.train()`` does.

        Args:
            subdir: Sub-directory of ``work_dir`` used as this run's
                ``output_dir`` (``"test"`` or ``"predict"``).
            batch: Batch size override, or ``None`` to use the test subset's
                configured ``batch_size``.
            kwargs: Additional ``transformers.TrainingArguments`` overrides.

        Returns:
            A trainer whose ``get_test_dataloader()`` is ready to use.

        Raises:
            ValueError: If the attached DataModule has no (non-empty) test split.
        """
        if self._datamodule is None:
            msg = "HFEngine._build_test_scoped_trainer() requires a DataModule; got a bare data-root path."
            raise ValueError(msg)

        test_subset = self._datamodule.subsets.get("test")
        if not test_subset:
            msg = "The attached DataModule has no (non-empty) test split."
            raise ValueError(msg)

        training_args_kwargs: dict[str, Any] = {
            "output_dir": str(self._work_dir / subdir),
            "per_device_eval_batch_size": batch or self._datamodule.test_subset.batch_size,
            "report_to": [],
            "use_cpu": self._device.type == "cpu",
        }
        training_args_kwargs.update(kwargs)

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self._device.index is not None:
            if world_size > 1:
                msg = (
                    "A single-device selection is for single-process runs. For torchrun, select devices through the "
                    "launcher's visible-device configuration; each process is assigned by LOCAL_RANK."
                )
                raise ValueError(msg)
            getattr(torch, self._device.type).set_device(self._device.index)

        args = TrainingArguments(**training_args_kwargs)
        if self._device.index is not None:
            args._setup_devices = self._device  # noqa: SLF001
            getattr(torch, self._device.type).set_device(self._device)

        return GetiTuneHFTrainer(
            self._model,
            self._datamodule,
            model=self._model.hf_model,
            args=args,
            eval_dataset=test_subset,
        )

    def export(
        self,
        checkpoint: PathLike | None = None,
        export_format: ExportFormat = ExportFormat.OPENVINO,
        export_precision: Precision = Precision.FP32,
        **kwargs,
    ) -> Path:
        """Export the model to OpenVINO IR or ONNX.

        Args:
            checkpoint: Optional checkpoint to load before exporting. When
                omitted, the currently loaded weights are used — typically
                ``best_checkpoint`` right after ``train()``.
            export_format: Target format.
            export_precision: Precision of the exported weights (FP32 or FP16).
            **kwargs: Reserved for future use; currently ignored.

        Returns:
            Path to the exported model file (``.xml`` for OpenVINO, ``.onnx``
            for ONNX).
        """
        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)
        elif self.best_checkpoint is not None:
            self._model.load_checkpoint(self.best_checkpoint)

        return self._model.export(
            output_dir=self._work_dir,
            base_name=self._EXPORTED_MODEL_BASE_NAME,
            export_format=export_format,
            precision=export_precision,
        )

    @classmethod
    def from_config(
        cls,
        config_path: PathLike,
        data: DataModule | PathLike | None = None,
        work_dir: PathLike | None = None,
        device: str | None = None,
        checkpoint: str | None = None,
        task: str | None = None,
        pretrained: bool | None = None,
        input_size: tuple[int, int] | None = None,
        overrides: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> HFEngine:
        """Build an engine from a Hugging Face recipe.

        Args:
            config_path: Recipe YAML containing ``backend: huggingface``.
            data: A pre-built :class:`DataModule` or a data-root path.
                Required.
            work_dir: Working directory for checkpoints and exports.
            device: Device to use.
            checkpoint: Optional warm-start checkpoint directory.
            pretrained: Override the recipe's model weight-loading setting.
            input_size: Override the recipe's model input size.
            task: Task type for disambiguation. Read from the recipe's
                top-level ``task`` field if not given here.
            overrides: Recipe section overrides using dot paths, applied before
                creating the DataModule, model, and training defaults.
            **kwargs: Backend-specific keyword arguments forwarded to
                :class:`HFEngine`.

        Returns:
            A fully configured :class:`HFEngine`.

        Raises:
            ValueError: If *data* is ``None``.
        """
        if data is None:
            msg = "data (a DataModule or data-root path) is required for HFEngine.from_config."
            raise ValueError(msg)

        configurator = Configurator(data=data, model=Path(str(config_path)), task=task)
        configurator.apply_overrides(overrides)

        datamodule = configurator.build_datamodule(input_size=input_size)
        label_info = datamodule.label_info
        model = configurator.create_model(label_info, pretrained=pretrained, input_size=input_size)

        engine_kwargs: dict[str, Any] = {**kwargs}
        if device is not None:
            engine_kwargs["device"] = device
        if checkpoint is not None:
            engine_kwargs["checkpoint"] = checkpoint

        return configurator.create_engine(model=model, data=datamodule, work_dir=work_dir, **engine_kwargs)

    @staticmethod
    def is_supported(model: MODEL, data: DATA) -> bool:
        """Return ``True`` when *model* is an :class:`HFModel`."""
        return bool(isinstance(model, HFModel) and isinstance(data, (DataModule, str, os.PathLike)))

    @property
    def work_dir(self) -> Path:
        """Working directory."""
        return self._work_dir

    @property
    def model(self) -> HFModel:
        """The wrapped :class:`HFModel`."""
        return self._model

    @property
    def datamodule(self) -> DATA:
        """The attached DataModule, or the data-root path.

        Raises:
            ValueError: If neither was configured.
        """
        if self._datamodule is not None:
            return self._datamodule
        if self._data_root is not None:
            return self._data_root
        msg = "No DataModule or data_root configured"
        raise ValueError(msg)

    @property
    def device(self) -> torch.device:
        """Resolved compute device."""
        return self._device

    @property
    def best_checkpoint(self) -> Path | None:
        """Canonical checkpoint after training.

        Unlike Lightning and Ultralytics, a Hugging Face checkpoint is a
        ``save_pretrained()`` directory rather than a file.
        """
        canonical = self._work_dir / self._CHECKPOINT_DIR_NAME
        return canonical if canonical.exists() else None

    @staticmethod
    def _resolve_accelerator(device: str | DeviceType) -> torch.device:
        """Resolve a device spec to a :class:`torch.device`.

        ``"auto"`` prefers XPU, then CUDA, then CPU, matching the rest of
        getitune.

        Args:
            device: ``"auto"``, ``"xpu"``, ``"cuda"``, ``"gpu"``, ``"cpu"``,
                or a :class:`DeviceType`.

        Returns:
            The resolved device.
        """
        if isinstance(device, DeviceType):
            device = {
                DeviceType.auto: "auto",
                DeviceType.xpu: "xpu",
                DeviceType.gpu: "cuda",
                DeviceType.cpu: "cpu",
            }.get(device, str(device.value))

        name = str(device).strip().lower()

        if name == "auto":
            if is_xpu_available():
                resolved_device = torch.device("xpu")
            elif torch.cuda.is_available():
                resolved_device = torch.device("cuda")
            else:
                resolved_device = torch.device("cpu")
        elif name in ("cuda", "gpu"):
            resolved_device = torch.device("cuda")
        else:
            resolved_device = torch.device(name)

        return resolved_device

    @staticmethod
    def _resolve_devices(
        devices: int | list[int] | None, device_type: str, default_num_devices: int = 1
    ) -> tuple[int, int | None]:
        """Resolve the Lightning-style device selection into a process count and optional device index."""
        if devices is None:
            return default_num_devices, 0 if default_num_devices == 1 else None
        if isinstance(devices, int):
            requested_num_devices = devices
            selected_device_index = 0 if devices == 1 else None
        elif isinstance(devices, list):
            if any(not isinstance(index, int) or isinstance(index, bool) for index in devices):
                msg = f"devices must contain integer indices, got {devices!r}"
                raise TypeError(msg)
            if any(index < 0 for index in devices):
                msg = f"device indices must be non-negative, got {devices!r}"
                raise ValueError(msg)
            if device_type not in ("cuda", "xpu"):
                msg = f"Explicit device indices are only supported for CUDA and XPU, got {device_type}"
                raise ValueError(msg)
            requested_num_devices = len(devices)
            selected_device_index = devices[0] if len(devices) == 1 else None
        else:
            msg = f"devices must be an integer, a list of indices, or None, got {devices!r}"
            raise TypeError(msg)

        if requested_num_devices < 1:
            msg = f"devices must select at least one device, got {devices!r}"
            raise ValueError(msg)
        return requested_num_devices, selected_device_index

    def __repr__(self) -> str:
        """Return a concise engine summary."""
        return (
            f"{type(self).__name__}(model={type(self._model).__name__}, "
            f"device={self._device}, work_dir={self._work_dir})"
        )
