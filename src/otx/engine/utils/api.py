"""OTX APIs for User-friendliness."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
import textwrap
from pathlib import Path

from otx.core.types.task import OTXTaskType
from otx.core.utils.imports import get_otx_root_path

RECIPE_PATH = get_otx_root_path() / "recipe"


def list_models(task: OTXTaskType | None = None, pattern: str | None = None, print_table: bool = False) -> list[str]:
    """Returns a list of available models for training.

    Args:
        task (OTXTaskType | None, optional): Recipe Filter by Task.
        pattern (Optional[str], optional): A string pattern to filter the list of available models. Defaults to None.
        print_table (bool, optional): Output the recipe information as a Rich.Table.
            This is primarily used for `otx find` in the CLI.

    Returns:
        list[str]: A list of available models for pretraining.

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
    task_type = OTXTaskType(task).name.lower() if task is not None else "**"
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
        table = Table(title="OTX Recipes", show_header=True, header_style="bold magenta")
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

    return list({Path(recipe).stem for recipe in recipe_list})
