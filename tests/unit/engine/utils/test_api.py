# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.core.types.task import OTXTaskType
from otx.engine.utils.api import RECIPE_PATH, list_models


def test_list_models() -> None:
    models = list_models()
    assert len(models) >= 32


@pytest.mark.parametrize("task", [task.value for task in OTXTaskType])
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
        "efficientnet_b0_semisl",
        "efficientnet_v2",
        "efficientnet_v2_semisl",
        "maskrcnn_efficientnetb2b",
        "maskrcnn_efficientnetb2b_tile",
        "tv_efficientnet_b3",
        "tv_efficientnet_b3_semisl",
        "tv_efficientnet_v2_l",
        "tv_efficientnet_v2_l_semisl",
    ]
    assert sorted(models) == sorted(target)


def test_list_models_print_table(capfd: pytest.CaptureFixture) -> None:
    list_models(pattern="efficient", print_table=True)

    out, _ = capfd.readouterr()
    assert "Task" in out
    assert "Model Name" in out
    assert "Recipe Path" in out
    assert "efficientnet_b0" in out
