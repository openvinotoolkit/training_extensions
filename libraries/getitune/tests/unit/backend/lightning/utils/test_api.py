# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from getitune.types.task import TaskType
from getitune.utils import RECIPE_PATH, list_models


def test_list_models() -> None:
    models = list_models()
    assert len(models) >= 32


@pytest.mark.parametrize("task", [task.value for task in TaskType])
def test_list_models_per_task(task: str) -> None:
    task_dir = task
    if task_dir.endswith("CLS"):
        task_dir = "classification/" + task_dir
    target_dir = RECIPE_PATH / task_dir.lower()
    target_recipes = [str(recipe.stem) for recipe in target_dir.glob("**/*.yaml")]

    models = list_models(task=task)
    assert sorted(models) == sorted(target_recipes)


def test_list_models_pattern() -> None:
    models = list_models(pattern="efficient")

    target = [
        "efficientnet_b0",
        "efficientnet_v2",
        "maskrcnn_efficientnetb2b",
        "maskrcnn_efficientnetb2b_tile",
        "efficientnet_b3",
    ]
    assert sorted(models) == sorted(target)


def test_list_models_print_table(capfd: pytest.CaptureFixture) -> None:
    list_models(pattern="efficient", print_table=True)

    out, _ = capfd.readouterr()
    assert "Task" in out
    assert "Model Name" in out
    assert "Recipe Path" in out
    assert "efficientnet_b0" in out
