# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configurator for Hugging Face recipes."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonargparse import ArgumentParser, Namespace

from getitune.backend.huggingface.models.base import HFModel
from getitune.backend.huggingface.tools.utils import (
    RECIPE_DIR,
    SUPPORTED_TASKS,
    TASK_TO_RECIPE_SUBDIR,
    build_subset_config,
    load_recipe,
)
from getitune.data.module import DataModule
from getitune.types.device import DeviceType
from getitune.types.label import LabelInfo
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from getitune.backend.huggingface.engine import HFEngine
    from getitune.types import PathLike


class Configurator:
    """Load a Hugging Face recipe and instantiate backend objects from it.

    Args:
        data: A data-root path (COCO/YOLO/Datumaro-native, whatever the
            underlying dataset format is) or an already-built
            :class:`~getitune.data.module.DataModule`.
        model: A recipe filename (``"rtdetrv2_r18"``), a full recipe path,
            or an already-instantiated :class:`HFModel`.
        task: Task identifier. Required when *model* is a bare name.
    """

    def __init__(
        self,
        data: PathLike | DataModule,
        model: str | PathLike | HFModel,
        *,
        task: str | TaskType | None = None,
    ) -> None:
        if isinstance(data, DataModule):
            self._data_root: Path | None = None
            self._datamodule: DataModule | None = data
        elif isinstance(data, (str, os.PathLike)):
            self._data_root = Path(data).resolve()
            self._datamodule = None
        else:
            msg = f"data must be PathLike or DataModule, got {type(data).__name__}"
            raise TypeError(msg)

        if task is not None:
            task_value = task.value if isinstance(task, TaskType) else str(task)
            if task_value not in SUPPORTED_TASKS:
                msg = f"Unsupported task '{task_value}'. Supported: {sorted(SUPPORTED_TASKS)}"
                raise ValueError(msg)
            self._task: str | None = task_value
        else:
            self._task = None

        self._model_config: dict[str, Any] | None = None
        self._model: HFModel | None = None
        self._data_config: dict[str, Any] | None = None
        self._training: dict[str, Any] = {}

        if isinstance(model, HFModel):
            self._model = model
        elif isinstance(model, (str, os.PathLike)):
            model_path = self._resolve_model_path(model)
            recipe = load_recipe(model_path)
            if self._task is None:
                recipe_task = recipe.get("task")
                if recipe_task is None:
                    msg = f"Recipe at {model_path} has no top-level 'task' field and none was provided."
                    raise ValueError(msg)
                self._task = str(recipe_task)
            self._model_config = self._extract_model_section(recipe, model_path)
            self._data_config = recipe.get("data")
            self._training = copy.deepcopy(recipe.get("training", {}))
        else:
            msg = f"model must be str, PathLike, or HFModel, got {type(model).__name__}"
            raise TypeError(msg)

    def _resolve_model_path(self, model_ref: str | PathLike) -> Path:
        """Resolve a bare recipe name or path to an absolute recipe file path."""
        model_str = str(model_ref)

        if isinstance(model_ref, Path) or os.sep in model_str:
            path = Path(model_ref).resolve()
            if not path.exists():
                msg = f"Recipe not found: {path}"
                raise FileNotFoundError(msg)
            return path

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
                f"Model name should match recipe filename (e.g., 'rtdetrv2_r18' for 'rtdetrv2_r18.yaml')"
            )
            raise FileNotFoundError(msg)
        return path

    @staticmethod
    def _extract_model_section(recipe: dict[str, Any], recipe_path: Path) -> dict[str, Any]:
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
    def training(self) -> dict[str, Any]:
        """Training configuration dict (``max_epochs``, ``learning_rate``, ...)."""
        return self._training

    def build_datamodule(
        self, data_root: PathLike | None = None, input_size: tuple[int, int] | None = None
    ) -> DataModule:
        """Build a DataModule from the recipe's data config.

        Args:
            data_root: Dataset directory. Falls back to the *data* passed to
                the constructor when omitted.
            input_size: Optional override for the recipe's model input size.

        Returns:
            A fully constructed :class:`DataModule`.

        Raises:
            ValueError: If no data config, task, or data root is available.
        """
        if self._datamodule is not None:
            return self._datamodule

        if self._data_config is None:
            msg = "No data config available. The model must resolve to a recipe file."
            raise ValueError(msg)
        if self._task is None:
            msg = "task is required to build a DataModule"
            raise ValueError(msg)

        root = data_root if data_root is not None else self._data_root
        if root is None:
            msg = "data_root is required. Pass it to build_datamodule() or to the constructor as data=."
            raise ValueError(msg)

        data_config = self._data_config
        if "input_size" not in data_config:
            msg = "data config is missing 'input_size'"
            raise ValueError(msg)
        configured_input_size = data_config["input_size"]
        resolved_input_size = input_size or (int(configured_input_size[0]), int(configured_input_size[1]))

        for subset_name in ("train", "val", "test"):
            key = f"{subset_name}_subset"
            if key not in data_config:
                msg = f"data config is missing '{key}'"
                raise ValueError(msg)

        self._datamodule = DataModule(
            task=TaskType(self._task),
            data_root=str(root),
            train_subset=build_subset_config(data_config, "train", resolved_input_size),
            val_subset=build_subset_config(data_config, "val", resolved_input_size),
            test_subset=build_subset_config(data_config, "test", resolved_input_size),
            input_size=resolved_input_size,
        )
        return self._datamodule

    def create_model(
        self,
        label_info: LabelInfo | int,
        pretrained: bool | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> HFModel:
        """Instantiate the configured Hugging Face model via jsonargparse."""
        if self._model is not None:
            return self._model

        model_config = copy.deepcopy(self._model_config)
        if model_config is None:
            msg = "Model config is not loaded. Ensure the recipe has a 'model' section."
            raise ValueError(msg)
        if isinstance(label_info, int):
            label_info = LabelInfo.from_num_classes(num_classes=label_info)

        # SegLabelInfo.as_dict() includes `ignore_index`, which the LabelInfoTypes
        # union HFModel.__init__ accepts (LabelInfo | int | list[str]) has no slot
        # for — jsonargparse rejects the dict outright. HFSemanticSegModel already
        # accepts `ignore_index` as its own constructor keyword (see G11), so route
        # it there instead of cramming it into label_info.
        label_info_dict = label_info.as_dict()
        ignore_index = label_info_dict.pop("ignore_index", None)
        init_args = model_config.setdefault("init_args", {})
        init_args["label_info"] = label_info_dict
        if pretrained is not None:
            init_args["pretrained"] = pretrained
        if input_size is not None:
            init_args["input_size"] = input_size
        if ignore_index is not None:
            init_args.setdefault("ignore_index", ignore_index)

        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(HFModel, "model", required=False, fail_untyped=False)
        model = model_parser.instantiate_classes(Namespace(model=model_config)).get("model")

        self._model = model
        return model

    def create_engine(
        self,
        model: HFModel,
        data: DataModule | PathLike | None = None,
        work_dir: PathLike | None = None,
        device: str | DeviceType = DeviceType.auto,
        **engine_kwargs: Any,  # noqa: ANN401
    ) -> HFEngine:
        """Instantiate the configured :class:`HFEngine`.

        Forwards the recipe's ``training:`` block as ``HFEngine(training=...)``
        so ``engine.train()`` picks up its defaults without the caller having
        to re-specify ``max_epochs``/``batch``/``learning_rate``/etc. by hand.
        """
        from getitune.backend.huggingface.engine import HFEngine as _HFEngine

        if data is None:
            if self._datamodule is not None:
                data = self._datamodule
            elif self._data_root is not None:
                data = self._data_root
            else:
                msg = "No data available. Pass data= to create_engine() or call build_datamodule() first."
                raise ValueError(msg)

        engine_kwargs.setdefault("training", self._training)
        engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
        return _HFEngine(model=model, data=data, work_dir=work_dir, device=device, **engine_kwargs)
