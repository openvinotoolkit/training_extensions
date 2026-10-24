# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared utility functions for Ultralytics recipe handling."""

from __future__ import annotations

import copy
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeAlias, cast

import yaml
from omegaconf import DictConfig, OmegaConf

from getitune.config.data import SamplerConfig, SubsetConfig
from getitune.types.task import TaskType

RECIPE_DIR: Path = Path(__file__).resolve().parents[3] / "recipe"
TASK_TO_RECIPE_SUBDIR: dict[str, str] = {
    TaskType.DETECTION.value: "detection",
    TaskType.INSTANCE_SEGMENTATION.value: "instance_segmentation",
    TaskType.MULTI_CLASS_CLS.value: "classification/multi_class_cls",
    TaskType.MULTI_LABEL_CLS.value: "classification/multi_label_cls",
    TaskType.SEMANTIC_SEGMENTATION.value: "semantic_segmentation",
}

SUPPORTED_TASKS: frozenset[str] = frozenset(
    (
        TaskType.DETECTION.value,
        TaskType.INSTANCE_SEGMENTATION.value,
        TaskType.MULTI_CLASS_CLS.value,
        TaskType.MULTI_LABEL_CLS.value,
        TaskType.SEMANTIC_SEGMENTATION.value,
    ),
)
ConfigValue: TypeAlias = str | int | float | bool | None | dict[str, "ConfigValue"] | list["ConfigValue"]


@contextmanager
def _temporary_resolver(name: str, func: Callable[..., Any]) -> Generator[None, None, None]:
    """Context manager to temporarily register an OmegaConf resolver.

    This ensures that the resolver is only active during the context,
    preventing global state pollution that could affect other parts
    of the system (e.g., jsonargparse in the Lightning backend).

    Args:
        name: Resolver name (e.g., "include").
        func: Resolver function.

    Yields:
        None
    """
    # Save the old resolver (if any)
    had_resolver = OmegaConf.has_resolver(name)
    old_resolver = OmegaConf._get_resolver(name) if had_resolver else None  # noqa: SLF001

    try:
        # Register the new resolver
        OmegaConf.register_new_resolver(name, func, replace=True)
        yield
    finally:
        # Restore the old state
        if had_resolver and old_resolver is not None:
            OmegaConf.register_new_resolver(name, old_resolver, replace=True)
        else:
            OmegaConf.clear_resolver(name)


def load_recipe(recipe_path: Path) -> dict[str, Any]:
    """Load a recipe with external data references and overrides, all at once.

    This function uses OmegaConf to handle:
    - External data file references (e.g., ``data: ../_base_/data/detection.yaml``)
    - Override sections (e.g., ``overrides.data``)
    - Deep merging of overrides into base config

    Args:
        recipe_path: Path to the recipe YAML file. Must exist.

    Returns:
        The fully resolved recipe dict with all references and overrides applied.

    Raises:
        FileNotFoundError: If *recipe_path* does not exist.
        TypeError: If the recipe is not a YAML mapping.
    """
    if not recipe_path.exists():
        msg = f"Recipe file not found: {recipe_path}"
        raise FileNotFoundError(msg)

    recipe_dir = recipe_path.parent

    def _include_yaml(filename: str) -> DictConfig:
        """Load a YAML file relative to the recipe directory, returning OmegaConf config."""
        path = (recipe_dir / filename).resolve()
        if not path.exists():
            msg = f"Referenced file not found: {path}"
            raise FileNotFoundError(msg)
        with open(path) as f:  # noqa: PTH123
            data = yaml.safe_load(f)
        return OmegaConf.create(data)

    with open(recipe_path) as f:  # noqa: PTH123
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        msg = f"Recipe must be a YAML mapping, got {type(raw).__name__}"
        raise TypeError(msg)

    # Convert string data references to OmegaConf interpolations
    if isinstance(raw.get("data"), str):
        raw["data"] = f"${{include:{raw['data']}}}"

    cfg = OmegaConf.create(raw)

    # Resolve interpolations first (so we can merge overrides into the actual data)
    with _temporary_resolver("include", _include_yaml):
        resolved = OmegaConf.to_container(cfg, resolve=True)
        cfg = cast("dict[str, Any]", resolved)

    # Apply all overrides via merge
    overrides = raw.get("overrides", {})
    if overrides:
        _deep_merge_inplace(cfg, overrides)

    # Remove overrides from final result
    cfg.pop("overrides", None)

    return cfg


def _deep_merge_inplace(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Deep-merge *overrides* into *base* in-place.

    This is used internally by load_recipe to apply overrides.
    """
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge_inplace(base[key], value)
        else:
            base[key] = value


def build_subset_config(
    data_config: dict[str, Any],
    subset_name: str,
    input_size: tuple[int, int],
) -> SubsetConfig:
    """Build a SubsetConfig from a data config dict entry."""
    key = f"{subset_name}_subset"
    subset_data = copy.deepcopy(data_config[key])
    subset_data["input_size"] = input_size

    sampler_data = subset_data.pop("sampler", None)
    if isinstance(sampler_data, dict):
        sampler = SamplerConfig(**sampler_data)
    elif sampler_data is None:
        sampler = SamplerConfig()
    else:
        sampler = sampler_data

    return SubsetConfig(sampler=sampler, **subset_data)


def flatten_overrides(overrides: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys."""
    flat: dict[str, Any] = {}
    for key, value in overrides.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten_overrides(value, full_key))
        else:
            flat[full_key] = value
    return flat
