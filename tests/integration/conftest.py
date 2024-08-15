# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
from otx.core.types.task import OTXTaskType


@pytest.fixture(scope="module", autouse=True)
def fxt_open_subprocess(request: pytest.FixtureRequest) -> bool:
    """Open subprocess for each CLI integration test case.

    This option can be used for easy memory management
    while running consecutive multiple tests (default: false).
    """
    return request.config.getoption("--open-subprocess", False)


def find_recipe_folder(base_path: Path, folder_name: str) -> Path:
    """
    Find the folder with the given name within the specified base path.

    Args:
        base_path (Path): The base path to search within.
        folder_name (str): The name of the folder to find.

    Returns:
        Path: The path to the folder.
    """
    for folder_path in base_path.rglob(folder_name):
        if folder_path.is_dir():
            return folder_path
    msg = f"Folder {folder_name} not found in {base_path}."
    raise FileNotFoundError(msg)


def get_task_list(task: str) -> list[OTXTaskType]:
    if task == "all":
        tasks = [task_type for task_type in OTXTaskType if task_type != OTXTaskType.DETECTION_SEMI_SL]
    elif task == "multi_cls_classification":
        tasks = [OTXTaskType.MULTI_CLASS_CLS]
    elif task == "multi_label_classification":
        tasks = [OTXTaskType.MULTI_LABEL_CLS]
    elif task == "hlabel_classification":
        tasks = [OTXTaskType.H_LABEL_CLS]
    elif task == "classification":
        tasks = [OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS]
    elif task == "action":
        tasks = [OTXTaskType.ACTION_CLASSIFICATION]
    elif task == "visual_prompting_all":
        tasks = [OTXTaskType.VISUAL_PROMPTING, OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING]
    elif task == "visual_prompting":
        tasks = [OTXTaskType.VISUAL_PROMPTING]
    elif task == "zero_shot_visual_prompting":
        tasks = [OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING]
    elif task == "anomaly":
        tasks = [OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION]
    elif task == "anomaly_classification":
        tasks = [OTXTaskType.ANOMALY_CLASSIFICATION]
    elif task == "anomaly_detection":
        tasks = [OTXTaskType.ANOMALY_DETECTION]
    elif task == "anomaly_segmentation":
        tasks = [OTXTaskType.ANOMALY_SEGMENTATION]
    else:
        tasks = [OTXTaskType(task.upper())]
    return tasks


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
    recipe_dir = [find_recipe_folder(recipe_path, task_type.value.lower()) for task_type in task_list]

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


@pytest.fixture(scope="session")
def fxt_asset_dir() -> Path:
    return Path(__file__).parent.parent / "assets"


# [TODO]: This is a temporary approach.
@pytest.fixture(scope="module")
def fxt_target_dataset_per_task() -> dict:
    return {
        "multi_class_cls": "tests/assets/classification_dataset",
        "multi_class_cls_semisl": "tests/assets/classification_semisl_dataset/unlabeled",
        "multi_label_cls": "tests/assets/multilabel_classification",
        "h_label_cls": "tests/assets/hlabel_classification",
        "detection": "tests/assets/car_tree_bug",
        "rotated_detection": "tests/assets/car_tree_bug",
        "instance_segmentation": "tests/assets/car_tree_bug",
        "semantic_segmentation": "tests/assets/common_semantic_segmentation_dataset/supervised",
        "semantic_segmentation_semisl": "tests/assets/common_semantic_segmentation_dataset/unlabeled",
        "action_classification": "tests/assets/action_classification_dataset/",
        "visual_prompting": "tests/assets/car_tree_bug",
        "zero_shot_visual_prompting": "tests/assets/car_tree_bug_zero_shot",
        "anomaly_classification": "tests/assets/anomaly_hazelnut",
        "anomaly_detection": "tests/assets/anomaly_hazelnut",
        "anomaly_segmentation": "tests/assets/anomaly_hazelnut",
    }


@pytest.fixture(scope="module")
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
        "visual_prompting": [],
        "zero_shot_visual_prompting": [],
        "anomaly_classification": [],
        "anomaly_detection": [],
        "anomaly_segmentation": [],
    }
