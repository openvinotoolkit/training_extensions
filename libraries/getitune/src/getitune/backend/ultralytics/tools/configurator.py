# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configurator for Ultralytics recipes."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

from jsonargparse import ArgumentParser, Namespace

from getitune.backend.ultralytics.models.base import UltralyticsModel
from getitune.backend.ultralytics.tools.utils import (
    RECIPE_DIR,
    SUPPORTED_TASKS,
    TASK_TO_RECIPE_SUBDIR,
    build_subset_config,
    flatten_overrides,
    load_recipe,
)
from getitune.config.data import TileConfig
from getitune.data.module import DataModule
from getitune.types.device import DeviceType
from getitune.types.label import LabelInfo
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from collections.abc import Mapping

    from getitune.backend.ultralytics.engine import UltralyticsEngine
    from getitune.types import PathLike


class Configurator:
    """Load Ultralytics recipes and instantiate backend objects.

    Args:
        data: One of:
            - ``Path`` / ``str``: YOLO data.yaml path / getitune data path (Datumaro supported formats).
            - ``DataModule``: A fully built DataModule instance.
        model: One of:
            - ``str``: Model variant name (e.g. ``"yolo26s"``, ``"yolo26s-seg"``)
              or recipe filename (e.g. ``"path/to/yolo26_s.yaml"``). Resolved against
              the recipes folder using the configured task.
            - ``Path``: Path to a model config YAML file. Loaded directly.
            - ``UltralyticsModel``: An already-instantiated model.
        task: Task identifier (``TaskType.DETECTION`` or
            ``TaskType.INSTANCE_SEGMENTATION``). Required when resolving a
            model name string.
        training: Optional training args dict (epochs, batch, lr0, etc.).
        export: Optional export config dict (format, precision, thresholds).
    """

    def __init__(
        self,
        data: PathLike | DataModule,
        model: str | PathLike | UltralyticsModel,
        *,
        task: str | TaskType | None = None,
        training: dict[str, Any] | None = None,
        export: dict[str, Any] | None = None,
    ) -> None:
        # ── Data ─────────────────────────────────────────────────────────
        if isinstance(data, (str, os.PathLike)):
            self._data_root: Path | None = Path(data).resolve()
            self._datamodule: DataModule | None = None
        elif isinstance(data, DataModule):
            self._data_root = None
            self._datamodule = data
        else:
            msg = f"data must be PathLike or DataModule, got {type(data).__name__}"
            raise TypeError(msg)

        # ── Task (validate first so model name resolution can use it) ───
        if task is not None:
            task_value = task.value if isinstance(task, TaskType) else str(task)
            if task_value not in SUPPORTED_TASKS:
                msg = f"Unsupported task '{task_value}'. Supported: {sorted(SUPPORTED_TASKS)}"
                raise ValueError(msg)
            self._task: str | None = task_value
        else:
            self._task = None

        # ── Model ────────────────────────────────────────────────────────
        self._model_config: dict[str, Any] | None = None
        self._model: UltralyticsModel | None = None
        self._data_config: dict[str, Any] | None = None
        self._training: dict[str, Any] = {}
        self._export: dict[str, Any] = {}

        if isinstance(model, UltralyticsModel):
            self._model = model
        elif isinstance(model, (str, os.PathLike)):
            model_path = self._resolve_model_path(model)
            recipe = load_recipe(model_path)
            self._model_config = self._extract_model_section(recipe, model_path)
            self._data_config = recipe.get("data")
            self._training = copy.deepcopy(recipe.get("training", {}))
            self._export = copy.deepcopy(recipe.get("export", {}))
        else:
            msg = f"model must be str, PathLike, or UltralyticsModel, got {type(model).__name__}"
            raise TypeError(msg)

        # ── Training / export: merge constructor overrides on top ────────
        if training:
            self._training.update(copy.deepcopy(training))
        if export:
            self._export.update(copy.deepcopy(export))

    def _resolve_model_path(self, model_ref: str | PathLike) -> Path:
        """Resolve a model name string or path to an absolute file path.

        Args:
            model_ref: Either a path to a recipe file, or a bare model name
                       (e.g., "yolo26_s" for recipe "yolo26_s.yaml").

        Returns:
            Absolute path to the recipe file.

        Raises:
            FileNotFoundError: If the recipe file doesn't exist.
            ValueError: If task is required but not set.
        """
        model_str = str(model_ref)

        # Case 1: Path object or string with path separators
        if isinstance(model_ref, Path) or os.sep in model_str:
            path = Path(model_ref).resolve()
            if not path.exists():
                msg = f"Recipe not found: {path}"
                raise FileNotFoundError(msg)
            return path

        # Case 2: Bare model name - resolve via task subdir
        if self._task is None:
            msg = f"Cannot resolve model name '{model_str}' without task=. Provide task= or pass a full path."
            raise ValueError(msg)

        subdir = TASK_TO_RECIPE_SUBDIR.get(self._task)
        if subdir is None:
            msg = f"No recipe subdir for task '{self._task}'"
            raise ValueError(msg)

        path = (RECIPE_DIR / subdir / f"{model_str}.yaml").resolve()
        if not path.exists():
            msg = (
                f"Recipe not found: {path}\n"
                f"Model name should match recipe filename (e.g., 'yolo26_s' for 'yolo26_s.yaml')"
            )
            raise FileNotFoundError(msg)
        return path

    @staticmethod
    def _extract_model_section(recipe: dict[str, Any], recipe_path: Path) -> dict[str, Any]:
        """Extract and validate the model section from a loaded recipe."""
        model_config = recipe.get("model")
        if not isinstance(model_config, dict):
            msg = f"Recipe at {recipe_path} is missing a valid 'model' section"
            raise TypeError(msg)

        if "class_path" not in model_config:
            msg = f"Recipe at {recipe_path} has a 'model' section without 'class_path'"
            raise ValueError(msg)

        return copy.deepcopy(model_config)

    @property
    def task(self) -> str | None:
        """Configured task identifier."""
        return self._task

    @property
    def datamodule(self) -> DataModule | None:
        """The DataModule if provided or built, otherwise ``None``."""
        return self._datamodule

    @property
    def data_root(self) -> Path | None:
        """Data root path if provided, otherwise ``None``."""
        return self._data_root

    @property
    def model(self) -> UltralyticsModel | None:
        """The UltralyticsModel if already built, otherwise ``None``."""
        if self._model is None:
            msg = "Model not instantiated yet. Call create_model() with label_info / num_classes to build it."
            warn(msg, stacklevel=2)
        return self._model

    @property
    def model_config(self) -> dict[str, Any] | None:
        """Stored model section from the recipe."""
        return self._model_config

    @property
    def training(self) -> dict[str, Any]:
        """Training configuration dict (epochs, batch, lr0, etc.)."""
        return self._training

    @property
    def export(self) -> dict[str, Any]:
        """Export configuration dict (format, precision, thresholds)."""
        return self._export

    @classmethod
    def _from_recipe_dict(cls, recipe: dict[str, Any], recipe_path: Path) -> Configurator:
        """Internal factory: create Configurator from already-loaded recipe dict.

        This avoids double-loading the YAML file when convert() has already
        read and validated the recipe.

        Args:
            recipe: The loaded recipe dict (already validated as a mapping, with
                data references and overrides resolved).
            recipe_path: Path to the recipe file (for resolving data references).

        Returns:
            A new Configurator instance.
        """
        instance = cls.__new__(cls)
        v = vars(instance)

        # Initialize all attributes
        v["_data_root"] = None
        v["_datamodule"] = None
        v["_model"] = None

        # Validate and set task
        task = recipe.get("task")
        if task not in SUPPORTED_TASKS:
            msg = f"Unsupported task '{task}'. Supported: {sorted(SUPPORTED_TASKS)}"
            raise ValueError(msg)
        v["_task"] = task

        # Extract sections from the already-loaded recipe
        v["_model_config"] = cls._extract_model_section(recipe, recipe_path)
        v["_data_config"] = recipe.get("data")
        v["_training"] = copy.deepcopy(recipe.get("training", {}))
        v["_export"] = copy.deepcopy(recipe.get("export", {}))

        return instance

    @classmethod
    def convert(cls, recipe_path: PathLike, hyper_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load an Ultralytics recipe, apply hyper-parameters, return config dict.

        This is the primary entry point for the application backend. It
        parses the recipe YAML and produces a flat config dict suitable
        for the application trainer.

        Args:
            recipe_path: Path to the recipe YAML file.
            hyper_parameters: Optional Geti-style hyper-parameters to apply.

        Returns:
            Flat config dict with ``backend``, ``task``, ``model``,
            ``data``, ``training``, ``export`` keys.
        """
        path = Path(recipe_path)
        if not path.exists():
            msg = f"Recipe file not found: {path}"
            raise FileNotFoundError(msg)

        # Load recipe once
        recipe = load_recipe(path)

        if recipe.get("backend") != "ultralytics":
            msg = f"Expected backend 'ultralytics', got '{recipe.get('backend')}'"
            raise ValueError(msg)

        # Use factory method to avoid double-loading
        configurator = cls._from_recipe_dict(recipe, path)

        if hyper_parameters:
            configurator.apply_hyper_parameters(hyper_parameters)

        return configurator.to_config_dict()

    def apply_hyper_parameters(self, hyper_parameters: dict[str, Any]) -> None:
        """Apply standard Geti hyper-parameters.

        Maps all backend-supported hyperparameters to Ultralytics training args:

        - ``learning_rate`` → ``lr0``
        - ``weight_decay`` → ``weight_decay``
        - ``max_epochs`` → ``epochs``
        - ``batch_size`` → ``batch``
        - ``num_workers`` → ``data.*_subset.num_workers``
        - ``input_size_height/width`` → ``imgsz``
        - ``early_stopping`` → ``patience`` (0 = disabled)
        - ``scheduler.warmup`` → ``warmup_epochs``
        - ``gradient_clip`` → ``max_grad_norm`` (used by trainer mixin)
        - ``gradient_accumulation`` → ``nbs`` (nominal batch size)
        """
        training = hyper_parameters.get("training", {})
        if not training:
            return

        if (lr := training.get("learning_rate")) is not None:
            self._training["lr0"] = float(lr)
        if (weight_decay := training.get("weight_decay")) is not None:
            self._training["weight_decay"] = float(weight_decay)
        if (max_epochs := training.get("max_epochs")) is not None:
            self._training["epochs"] = int(max_epochs)
        if (batch_size := training.get("batch_size")) is not None:
            self._training["batch"] = int(batch_size)
            if self._data_config is not None:
                for subset in ("train_subset", "val_subset", "test_subset"):
                    if subset in self._data_config:
                        self._data_config[subset]["batch_size"] = int(batch_size)
        if (num_workers := training.get("num_workers")) is not None and self._data_config is not None:
            for subset in ("train_subset", "val_subset", "test_subset"):
                if subset in self._data_config:
                    self._data_config[subset]["num_workers"] = int(num_workers)

        height = training.get("input_size_height")
        width = training.get("input_size_width")
        if height is not None and width is not None:
            h, w = int(height), int(width)
            if self._model_config is not None:
                self._model_config.setdefault("init_args", {})["imgsz"] = max(h, w)

        early_stopping = training.get("early_stopping")
        if isinstance(early_stopping, dict):
            if early_stopping.get("enable") is False:
                self._training["patience"] = 0
            elif early_stopping.get("enable") is True and early_stopping.get("patience") is not None:
                self._training["patience"] = int(early_stopping["patience"])

        self._apply_scheduler(training.get("scheduler"), self._training)
        self._apply_gradient_clip(training.get("gradient_clip"), self._training)
        self._apply_gradient_accumulation(training.get("gradient_accumulation"), self._training)

    @staticmethod
    def _apply_scheduler(scheduler_cfg: dict[str, Any] | None, train_cfg: dict[str, Any]) -> None:
        """Map scheduler parameters to Ultralytics training args."""
        if not isinstance(scheduler_cfg, dict):
            return

        warmup = scheduler_cfg.get("warmup")
        if isinstance(warmup, dict):
            if warmup.get("enable") and warmup.get("epochs") is not None:
                train_cfg["warmup_epochs"] = float(warmup["epochs"])
            elif warmup.get("enable") is False:
                train_cfg["warmup_epochs"] = 0.0

    @staticmethod
    def _apply_gradient_clip(gradient_clip_cfg: dict[str, Any] | None, train_cfg: dict[str, Any]) -> None:
        """Map gradient clipping to Ultralytics training args."""
        if not isinstance(gradient_clip_cfg, dict):
            return

        if gradient_clip_cfg.get("enable"):
            max_norm = gradient_clip_cfg.get("max_grad_norm")
            if max_norm is not None:
                train_cfg["max_grad_norm"] = float(max_norm)
        else:
            train_cfg["max_grad_norm"] = 0.0

    @staticmethod
    def _apply_gradient_accumulation(gradient_accum_cfg: dict[str, Any] | None, train_cfg: dict[str, Any]) -> None:
        """Map gradient accumulation to Ultralytics ``nbs`` parameter."""
        if not isinstance(gradient_accum_cfg, dict):
            return

        batch_size = train_cfg.get("batch", 16)
        if gradient_accum_cfg.get("enable"):
            batches = gradient_accum_cfg.get("batches", 1)
            if batches is not None and int(batches) > 1:
                train_cfg["nbs"] = int(batch_size) * int(batches)
            else:
                train_cfg["nbs"] = int(batch_size)
        else:
            train_cfg["nbs"] = int(batch_size)

    def to_config_dict(self) -> dict[str, Any]:
        """Return the config consumed by the application trainer.

        Places ``max_epochs`` at the top level so the application trainer
        can access it uniformly regardless of backend.
        """
        config: dict[str, Any] = {}
        config["backend"] = "ultralytics"
        config["task"] = self._task

        if self._model_config is not None:
            config["model"] = copy.deepcopy(self._model_config)
        elif self._model is not None:
            config["model"] = {
                "class_path": f"{type(self._model).__module__}.{type(self._model).__qualname__}",
                "init_args": {
                    "model_name": self._model.model_name,
                    "pretrained": self._model.pretrained,
                    "imgsz": self._model.imgsz,
                },
            }

        if self._data_config is not None:
            data = copy.deepcopy(self._data_config)
            data.setdefault("tile_config", {"enable_tiler": False, "enable_adaptive_tiling": False})
            config["data"] = data
        elif self._datamodule is not None:
            config["data"] = self._reconstruct_data_config(self._datamodule)

        if self._training:
            config["training"] = copy.deepcopy(self._training)
            if "epochs" in self._training:
                config["max_epochs"] = self._training["epochs"]

        if self._export:
            config["export"] = copy.deepcopy(self._export)

        return config

    def apply_overrides(self, overrides: Mapping[str, Any] | None = None) -> None:
        """Merge supported dot-path overrides into the config."""
        if not overrides:
            return
        for key, value in flatten_overrides(overrides).items():
            self._set_dot_path(key, value)

    def _set_dot_path(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Set a dot-path key on the appropriate internal store.

        Args:
            key: Dot-separated path like ``"training.epochs"``.
            value: Value to set.

        Raises:
            ValueError: If *key* has fewer than two parts.
            KeyError: If the section in *key* is not available.
        """
        parts = key.split(".")
        if len(parts) < 2:
            msg = f"Override key must be a dot path, got '{key}'"
            raise ValueError(msg)

        section = parts[0]
        rest = parts[1:]

        if section == "training":
            self._set_nested(self._training, rest, value)
        elif section == "export":
            self._set_nested(self._export, rest, value)
        elif section == "model" and self._model_config is not None:
            self._set_nested(self._model_config, rest, value)
        elif section == "data" and self._data_config is not None:
            self._set_nested(self._data_config, rest, value)
        else:
            msg = f"Cannot set override '{key}': section '{section}' is not available in this configurator"
            raise KeyError(msg)

    @staticmethod
    def _set_nested(
        d: dict[str, Any],
        parts: list[str],
        value: Any,  # noqa: ANN401
    ) -> None:
        """Set a nested key path in a dict, creating intermediate dicts as needed."""
        current = d
        for part in parts[:-1]:
            next_value = current.setdefault(part, {})
            if not isinstance(next_value, dict):
                msg = f"Cannot set '{'.'.join(parts)}': '{part}' is not a mapping"
                raise TypeError(msg)
            current = next_value
        current[parts[-1]] = value

    def build_datamodule(self, data_root: PathLike | None = None) -> DataModule:
        """Build a DataModule from the stored data config.

        If data was already a DataModule (passed to the constructor), it is
        returned as-is.  Otherwise a fresh DataModule is constructed from
        the recipe's resolved data config.

        Args:
            data_root: Path to the dataset directory (COCO-style).  When
                omitted, the ``data`` path passed to the constructor is used.

        Returns:
            A fully constructed :class:`DataModule`.

        Raises:
            ValueError: When no data config is available, no *data_root* is
                provided, or required keys are missing from the data config.
        """
        if self._datamodule is not None:
            return self._datamodule

        if self._data_config is None:
            msg = "No data config available. The model must resolve to a recipe file."
            raise ValueError(msg)

        if self._task is None:
            msg = "task is required to build DataModule"
            raise ValueError(msg)

        root = data_root if data_root is not None else self._data_root
        if root is None:
            msg = "data_root is required. Pass it to build_datamodule() or to the constructor as data=."
            raise ValueError(msg)

        data_config = self._data_config
        if "input_size" not in data_config:
            msg = "data config is missing 'input_size'"
            raise ValueError(msg)
        input_size_raw = data_config["input_size"]
        if not isinstance(input_size_raw, (list, tuple)):
            msg = f"input_size must be list or tuple, got {type(input_size_raw)}"
            raise TypeError(msg)
        if len(input_size_raw) != 2:
            msg = f"input_size must have 2 elements, got {len(input_size_raw)}"
            raise ValueError(msg)
        input_size = (int(input_size_raw[0]), int(input_size_raw[1]))

        for subset_name in ("train", "val", "test"):
            key = f"{subset_name}_subset"
            if key not in data_config:
                msg = f"data config is missing '{key}'"
                raise ValueError(msg)

        train_subset = build_subset_config(data_config, "train", input_size)
        val_subset = build_subset_config(data_config, "val", input_size)
        test_subset = build_subset_config(data_config, "test", input_size)

        self._datamodule = DataModule(  # type: ignore[has-type]
            task=TaskType(self._task),
            data_root=str(root),
            train_subset=train_subset,
            val_subset=val_subset,
            test_subset=test_subset,
            tile_config=TileConfig(enable_tiler=False),
            input_size=input_size,
        )
        return self._datamodule

    def create_model(self, label_info: LabelInfo | int, weights_path: PathLike | None = None) -> UltralyticsModel:
        """Instantiate the configured Ultralytics model via jsonargparse."""
        if self._model is not None:
            if weights_path is not None:
                self._model.load_checkpoint(weights_path)
            return self._model

        model_config = copy.deepcopy(self._model_config)  # type: ignore[union-attr]
        if model_config is None:
            msg = "Model config is not loaded. Ensure the recipe has a 'model' section."
            raise ValueError(msg)
        if isinstance(label_info, int):
            label_info = LabelInfo.from_num_classes(num_classes=label_info)
        model_config.setdefault("init_args", {})["label_info"] = label_info.as_dict()

        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(UltralyticsModel, "model", required=False, fail_untyped=False)
        model = model_parser.instantiate_classes(Namespace(model=model_config)).get("model")

        if weights_path is not None:
            model.load_checkpoint(weights_path)

        self._model = model
        return model

    def create_engine(
        self,
        model: UltralyticsModel,
        data: DataModule | PathLike | None = None,
        work_dir: PathLike | None = None,
        device: str | DeviceType = DeviceType.auto,
        **engine_kwargs,
    ) -> UltralyticsEngine:
        """Instantiate the configured Ultralytics engine."""
        from getitune.backend.ultralytics.engine import UltralyticsEngine as _UltralyticsEngine

        if data is None:
            if self._datamodule is not None:
                data = self._datamodule
            elif self._data_root is not None:
                data = self._data_root
            else:
                msg = "No data available. Pass data= to create_engine() or call build_datamodule() first."
                raise ValueError(msg)

        return _UltralyticsEngine(
            model=model,
            data=data,
            work_dir=work_dir,
            device=device,
            train_args=copy.deepcopy(self._training),
            export_args={
                "confidence_threshold": self._export.get("confidence_threshold", 0.25),
                "iou_threshold": self._export.get("iou_threshold", 0.5),
            },
            **engine_kwargs,
        )

    @staticmethod
    def _reconstruct_data_config(datamodule: DataModule) -> dict[str, Any]:
        """Build a data config dict from a live DataModule for to_config_dict()."""
        config: dict[str, Any] = {
            "input_size": list(datamodule.input_size) if datamodule.input_size else None,
            "tile_config": {
                "enable_tiler": datamodule.tile_config.enable_tiler,
                "enable_adaptive_tiling": datamodule.tile_config.enable_adaptive_tiling,
            },
        }
        for name, subset_cfg in [
            ("train", datamodule.train_subset),
            ("val", datamodule.val_subset),
            ("test", datamodule.test_subset),
        ]:
            config[f"{name}_subset"] = {
                "batch_size": subset_cfg.batch_size,
                "subset_name": subset_cfg.subset_name,
                "augmentations_cpu": copy.deepcopy(subset_cfg.augmentations_cpu),
                "augmentations_gpu": copy.deepcopy(subset_cfg.augmentations_gpu),
                "num_workers": subset_cfg.num_workers,
                "input_size": list(subset_cfg.input_size) if subset_cfg.input_size else None,
            }
        return config
