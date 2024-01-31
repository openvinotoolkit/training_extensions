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
    if task.endswith("CLS"):
        task = "classification/" + task
    elif task.startswith("ACTION"):
        task = "action/" + task
    target_dir = RECIPE_PATH / task.lower()
    target_recipes = [str(recipe.stem) for recipe in target_dir.glob("**/*.yaml")]

    models = list_models(task=task)
    assert sorted(models) == sorted(target_recipes)


def test_list_models_pattern() -> None:
    models = list_models(pattern="efficient")

    target = [
        "efficientnet_v2_light",
        "efficientnet_b0_light",
        "maskrcnn_efficientnetb2b",
        "otx_efficientnet_v2",
        "otx_efficientnet_b0",
    ]
    assert sorted(models) == sorted(target)


def test_list_models_print_table(capfd: pytest.CaptureFixture) -> None:
    list_models(pattern="otx_efficient", print_table=True)

    out, _ = capfd.readouterr()
    assert "Task" in out
    assert "Model Name" in out
    assert "Recipe Path" in out
    assert "otx_efficientnet_b0" in out
    assert "otx_efficientnet_v2" in out
