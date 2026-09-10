# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for creating engines from configurations and model names."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from getitune.engine.engine import Engine

if TYPE_CHECKING:
    from getitune.types import PathLike
    from getitune.types.task import TaskType
    from getitune.types.types import DATA, MODEL

# Recipe directory — parallel to the package root.
_RECIPE_PATH: Path = Path(__file__).parent.parent.parent / "recipe"
_RECIPE_SUFFIXES: frozenset[str] = frozenset({".yaml", ".yml"})
_WEIGHT_SUFFIXES: frozenset[str] = frozenset({".xml", ".onnx"})


def _resolve_recipe(model: MODEL, task: TaskType | str | None) -> Path:
    """Resolve a model name or YAML path to an absolute recipe path.

    Args:
        model: A YAML config path or a bare model name (e.g. ``"yolox_s"``).
        task: Optional task used to disambiguate when a model name matches
            recipes under multiple tasks.

    Returns:
        Absolute path to the resolved recipe file.

    Raises:
        FileNotFoundError: If the recipe file or model name cannot be resolved.
        ValueError: If *model* is a name that matches multiple tasks and *task*
            is not provided.
    """
    path = Path(str(model))

    # Explicit YAML path — use directly.
    if path.suffix in _RECIPE_SUFFIXES:
        if not path.exists():
            msg = f"Recipe file not found: {path}"
            raise FileNotFoundError(msg)
        return path.resolve()

    # Bare model name — search the recipe tree.
    name = path.stem if path.suffix else path.name
    matches = sorted([*_RECIPE_PATH.glob(f"**/{name}.yaml"), *_RECIPE_PATH.glob(f"**/{name}.yml")])

    if not matches:
        msg = (
            f"No recipe found for model '{name}' under {_RECIPE_PATH}. Check the model name or pass a full recipe path."
        )
        raise FileNotFoundError(msg)

    if len(matches) == 1:
        return matches[0]

    # Multiple matches — narrow by task if provided.
    if task is not None:
        task_str = task.value.lower() if hasattr(task, "value") else str(task).lower()
        task_matches = [m for m in matches if task_str in m.parts]
        if len(task_matches) == 1:
            return task_matches[0]
        if task_matches:
            matches = task_matches  # still ambiguous — fall through to error

    candidates = [str(m.relative_to(_RECIPE_PATH)) for m in matches]
    msg = f"Model name '{name}' matches multiple recipes: {candidates}. Pass task= to disambiguate."
    raise ValueError(msg)


def _read_backend(recipe_path: Path) -> str:
    """Read the ``backend`` field from a recipe YAML.

    Falls back to ``'lightning'`` when the field is absent — the convention
    for all Lightning-backed recipes.

    Args:
        recipe_path: Absolute path to the recipe YAML file.

    Returns:
        Backend name string (e.g. ``'lightning'``, ``'ultralytics'``, ``'huggingface'``).
    """
    with recipe_path.open() as fh:
        raw = yaml.safe_load(fh)
    return (raw or {}).get("backend") or "lightning"


def create_engine(
    model: MODEL,
    data: DATA,
    work_dir: PathLike | None = None,
    device: str | None = None,
    checkpoint: str | None = None,
    task: TaskType | str | None = None,
    **kwargs,
) -> Engine:
    """Create an engine.

    Accepts three forms for *model*:

    * **Model instance** (``LightningModel``, ``UltralyticsModel``, ``HFModel``,
      ``OVModel``) or a **weights path** (``.xml``, ``.onnx``) — for OpenVINO
      and ONNX models

    * **Recipe path** — a ``.yaml`` / ``.yml`` file.  The ``backend`` field in
      the recipe selects the engine; defaults to ``lightning`` when absent.

    * **Model name** — a bare string (e.g. ``"yolox_s"``) that is resolved to a
      recipe by searching the ``recipe/`` tree.  Pass ``task=`` when the name
      matches recipes under multiple tasks.

    Args:
        model: The model to use — instance, weights path, recipe path, or model name.
        data: DataModule or filesystem data-root path.
        work_dir: Working directory for checkpoints, exports, and logs.
            Defaults to ``"./getitune-workspace"``.
        device: Device to use (e.g., ``"auto"``, ``"xpu"``, ``"cpu"``, ``"gpu"``).
            Defaults to backend's default (auto).
        checkpoint: Optional path to a checkpoint to load model weights from
            before training (for pretrained or warm-start weights).
        task: Task type for disambiguation when a model name matches recipes
            under multiple tasks. Optional.
        **kwargs: Additional backend-specific keyword arguments
            (e.g., ``train_args``, ``export_args`` for Ultralytics;
            Trainer parameters for Lightning).

    Returns:
        An :class:`Engine` subclass instance.

    Raises:
        FileNotFoundError: If a recipe / model name cannot be resolved or
            if a YAML path does not exist on disk.
        ValueError: If a model name is ambiguous, the backend is unknown,
            or no engine supports the given model/data pair.
    """
    from getitune.backend.huggingface.engine import HFEngine
    from getitune.backend.lightning.engine import LightningEngine
    from getitune.backend.openvino.engine import OVEngine

    backend_to_engine: dict[str, type[Engine]] = {
        "lightning": LightningEngine,
        "openvino": OVEngine,
        "huggingface": HFEngine,
    }
    try:
        from getitune.backend.ultralytics.engine import UltralyticsEngine

        backend_to_engine["ultralytics"] = UltralyticsEngine
    except ImportError:
        pass

    # All known engine classes for the instance/path dispatch path,
    # including any dynamically registered custom subclasses.
    supported_engines: list[type[Engine]] = list(backend_to_engine.values())
    for child_engine in Engine.__subclasses__():
        if child_engine not in supported_engines:
            supported_engines.append(child_engine)

    # Build kwargs dict with common arguments (skip None values so defaults apply).
    common_kwargs: dict[str, Any] = {}
    if work_dir is not None:
        common_kwargs["work_dir"] = work_dir
    if device is not None:
        common_kwargs["device"] = device
    if checkpoint is not None:
        common_kwargs["checkpoint"] = checkpoint
    if task is not None:
        common_kwargs["task"] = task
    # Merge with user-supplied kwargs (user kwargs take precedence).
    common_kwargs.update(kwargs)

    # Classify the model argument.
    is_recipe_input = False
    if isinstance(model, (str, os.PathLike)):
        path = Path(str(model))
        if path.suffix in _RECIPE_SUFFIXES:
            # Explicit YAML config path — must exist on disk.
            if not path.exists():
                msg = f"Recipe file not found: {path}"
                raise FileNotFoundError(msg)
            is_recipe_input = True
        elif path.suffix not in _WEIGHT_SUFFIXES and not path.exists():
            # No weight suffix and no file on disk → treat as a model name.
            is_recipe_input = True

    # Recipe / model-name route.
    if is_recipe_input:
        recipe_path = _resolve_recipe(model, task)
        backend = _read_backend(recipe_path)

        engine_cls = backend_to_engine.get(backend)
        if engine_cls is None:
            msg = (
                f"Unknown backend '{backend}' declared in '{recipe_path}'. Known backends: {sorted(backend_to_engine)}"
            )
            raise ValueError(msg)

        return engine_cls.from_config(recipe_path, data, **common_kwargs)

    # Model instance or weights path.
    for engine_cls in supported_engines:
        if not hasattr(engine_cls, "is_supported"):
            msg = f"Engine {engine_cls.__name__} does not implement is_supported."
            raise ValueError(msg)
        if engine_cls.is_supported(model, data):
            return engine_cls(model=model, data=data, **common_kwargs)  # pyrefly: ignore[unexpected-keyword]

    msg = f"No engine found for model {model!r} and data {data!r}"
    raise ValueError(msg)
