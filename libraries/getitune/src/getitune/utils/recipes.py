# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for discovering and listing model recipes."""

from __future__ import annotations

import fnmatch
import importlib
import inspect
import textwrap
from pathlib import Path

from getitune.types.task import TaskType


def get_getitune_root_path() -> Path:
    """Return the root path of the getitune module.

    Returns:
        str: The root path of the getitune module.

    Raises:
        ModuleNotFoundError: If the getitune module is not found.
    """
    getitune_module = importlib.import_module("getitune")
    if getitune_module:
        file_path = inspect.getfile(getitune_module)
        return Path(file_path).parent
    msg = "Cannot find getitune."
    raise ModuleNotFoundError(msg)


RECIPE_PATH = get_getitune_root_path() / "recipe"


def list_models(
    task: TaskType | None = None, pattern: str | None = None, print_table: bool = False, return_recipes: bool = False
) -> list[str]:
    """Returns a list of available models for training.

    Args:
        task (TaskType | None, optional): Recipe Filter by Task.
        pattern (Optional[str], optional): A string pattern to filter the list of available models. Defaults to None.
        print_table (bool, optional): Output the recipe information as a Rich.Table.
            This is primarily used for ``getitune find`` in the CLI.
        return_recipes (bool, optional): If True, return the recipe paths instead of model names.

    Returns:
        list[str]: A list of available models or recipes for fine-tuning.


    Example:
        # Return all available model list.
        >>> models = list_models()
        >>> models
        ['atss_mobilenetv2', 'atss_r50_fpn', ...]

        # Return INSTANCE_SEGMENTATION model list.
        >>> models = list_models(task="INSTANCE_SEGMENTATION")
        >>> models
        ['maskrcnn_efficientnetb2b', 'maskrcnn_r50', 'maskrcnn_swint', 'openvino_model']

        # Return all available model list that matches the pattern.
        >>> models = list_models(task="MULTI_CLASS_CLS", pattern="*efficient")
        >>> models
        ['efficientnet_v2', 'efficientnet_b0', ...]

        # Print the recipe information as a Rich.Table (include task, model name, recipe path)
        >>> models = list_models(task="MULTI_CLASS_CLS", pattern="*efficient", print_table=True)
    """
    task_type = TaskType(task).name.lower() if task is not None else "**"
    recipe_list = [
        str(recipe) for recipe in RECIPE_PATH.glob(f"**/{task_type}/**/*.yaml") if "_base_" not in recipe.parts
    ]

    if pattern is not None:
        # Always match keys with any postfix.
        recipe_list = list(set(fnmatch.filter(recipe_list, f"*{pattern}*")))

    if print_table:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="getitune Recipes", show_header=True, header_style="bold magenta")
        table.add_column("Task")
        table.add_column("Model Name")
        table.add_column("Recipe Path")
        for recipe in recipe_list:
            recipe_path = (
                textwrap.fill(recipe, width=int(console.width / 2)) if len(recipe) > console.width / 2 else recipe
            )
            table.add_row(
                recipe.split("/")[-2].upper(),
                Path(recipe).stem,
                recipe_path,
            )
        console.print(table, width=console.width, justify="center")

    if return_recipes:
        return recipe_list

    return list({Path(recipe).stem for recipe in recipe_list})
