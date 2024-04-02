# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path

import pytest
from otx.core.types.task import OTXTaskType

from tests.test_helpers import find_folder


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--open-subprocess",
        action="store_true",
        help="Open subprocess for each CLI test case. "
        "This option can be used for easy memory management "
        "while running consecutive multiple tests (default: false).",
    )
    parser.addoption(
        "--task",
        action="store",
        default="all",
        type=str,
        help="Task type of OTX to use test.",
    )


@pytest.fixture(scope="session")
def fxt_ci_data_root() -> Path:
    data_root = Path(os.environ.get("CI_DATA_ROOT", "/home/validation/data/v2"))
    if not Path.is_dir(data_root):
        msg = f"cannot find {data_root}"
        raise FileNotFoundError(msg)
    return data_root


@pytest.fixture(scope="module", autouse=True)
def fxt_open_subprocess(request: pytest.FixtureRequest) -> bool:
    """Open subprocess for each CLI test case.

    This option can be used for easy memory management
    while running consecutive multiple tests (default: false).
    """
    return request.config.getoption("--open-subprocess", False)


def get_task_list(task: str) -> list[OTXTaskType]:
    if task == "all":
        return [task_type for task_type in OTXTaskType if task_type != OTXTaskType.DETECTION_SEMI_SL]
    if task == "classification":
        return [OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS]
    if task == "action":
        return [OTXTaskType.ACTION_CLASSIFICATION, OTXTaskType.ACTION_DETECTION]
    if task == "visual_prompting":
        return [OTXTaskType.VISUAL_PROMPTING, OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING]
    if task == "anomaly":
        return [OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION]
    return [OTXTaskType(task.upper())]


def pytest_configure(config):
    """Configure pytest options and set task, recipe, and recipe_ov lists.

    Args:
        config (pytest.Config): The pytest configuration object.

    Returns:
        None
    """
    task = config.getoption("--task")

    # This assumes have OTX installed in environment.
    otx_module = importlib.import_module("otx")
    # Modify RECIPE_PATH based on the task
    recipe_path = Path(inspect.getfile(otx_module)).parent / "recipe"
    task_list = get_task_list(task.lower())
    recipe_dir = [find_folder(recipe_path, task_type.value.lower()) for task_type in task_list]

    # Update RECIPE_LIST
    target_recipe_list = []
    target_ov_recipe_list = []
    for task_recipe_dir in recipe_dir:
        recipe_list = [str(p) for p in task_recipe_dir.glob("**/*.yaml") if "_base_" not in p.parts]
        recipe_ov_list = [str(p) for p in task_recipe_dir.glob("**/openvino_model.yaml") if "_base_" not in p.parts]
        recipe_list = set(recipe_list) - set(recipe_ov_list)

        target_recipe_list.extend(recipe_list)
        target_ov_recipe_list.extend(recipe_ov_list)
    tile_recipe_list = [recipe for recipe in target_recipe_list if "tile" in recipe]

    pytest.TASK_LIST = task_list
    pytest.RECIPE_LIST = target_recipe_list
    pytest.RECIPE_OV_LIST = target_ov_recipe_list
    pytest.TILE_RECIPE_LIST = tile_recipe_list


# [TODO]: This is a temporary approach.
@pytest.fixture()
def fxt_target_dataset_per_task(fxt_ci_data_root) -> dict:
    return {
        "multi_class_cls": Path(fxt_ci_data_root / "multiclass_classification/multiclass_CUB_small/1"),
        "multi_label_cls": Path(fxt_ci_data_root / "multilabel_classification/multilabel_CUB_small/1"),
        "h_label_cls": Path(fxt_ci_data_root / "hlabel_classification/hlabel_CUB_small/1"),
        "detection": Path(fxt_ci_data_root / "detection/pothole_small/1"),
        "rotated_detection": Path(fxt_ci_data_root / "detection/pothole_small/1"),
        "instance_segmentation": Path(fxt_ci_data_root / "instance_seg/wgisd_small/1"),
        "semantic_segmentation": Path(fxt_ci_data_root / "semantic_seg/kvasir_small/1"),
        "action_classification": Path(fxt_ci_data_root / "action/action_classification/ucf_kinetics_5percent_small"),
        "action_detection": Path(fxt_ci_data_root / "action/action_detection/UCF101_ava_5percent"),
        "visual_prompting": Path(fxt_ci_data_root / "visual_prompting/wgisd_small/1"),
        "zero_shot_visual_prompting": Path(
            fxt_ci_data_root / "zero_shot_visual_prompting/coco_car_person_medium_datumaro",
        ),
        "anomaly_classification": Path(fxt_ci_data_root / "anomaly/bottle_small/1"),
        "anomaly_detection": Path(fxt_ci_data_root / "anomaly/hazelnut_large"),
        "anomaly_segmentation": Path(fxt_ci_data_root / "anomaly/hazelnut_large"),
    }


@pytest.fixture()
def fxt_cli_override_command_per_task() -> dict:
    return {
        "multi_class_cls": [],
        "multi_label_cls": [],
        "h_label_cls": [],
        "detection": [],
        "rotated_detection": [],
        "instance_segmentation": [],
        "semantic_segmentation": [],
        "action_classification": [],
        "action_detection": [
            "--model.topk",
            "3",
        ],
        "visual_prompting": [],
        "zero_shot_visual_prompting": [],
        "anomaly_classification": [],
        "anomaly_detection": [],
        "anomaly_segmentation": [],
    }
