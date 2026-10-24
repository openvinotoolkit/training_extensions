# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from getitune.types.task import TaskType
from getitune.utils import get_getitune_root_path

RECIPE_PATH = get_getitune_root_path() / "recipe"

# Recipes selected for category-only integration runs (speed / balance / accuracy).
# This mirrors the TEMPLATE_ID_MAPPING kept in the application backend
# (app.execution.common.geti_config_converter) — only the non-ACTIVE entries.
CATEGORY_RECIPES_PER_TASK: dict[str, list[Path]] = {
    "multi_class_cls": [
        RECIPE_PATH / "classification" / "multi_class_cls" / "mobilenet_v3_large.yaml",  # speed
        RECIPE_PATH / "classification" / "multi_class_cls" / "vit_tiny.yaml",  # balance
        RECIPE_PATH / "classification" / "multi_class_cls" / "dino_v2.yaml",  # accuracy
    ],
    "detection": [
        RECIPE_PATH / "detection" / "yolox_s.yaml",  # speed
        RECIPE_PATH / "detection" / "deim_dfine_m.yaml",  # balance
        RECIPE_PATH / "detection" / "deim_dfine_l.yaml",  # accuracy
    ],
    "instance_segmentation": [
        RECIPE_PATH / "instance_segmentation" / "rfdetr_seg_small.yaml",  # speed
        RECIPE_PATH / "instance_segmentation" / "rfdetr_seg_medium.yaml",  # balance
        RECIPE_PATH / "instance_segmentation" / "rfdetr_seg_xlarge.yaml",  # accuracy
    ],
}


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


def get_task_list(task: str) -> list[TaskType]:
    if task == "all":
        tasks = list(TaskType)
    elif task == "multi_class_cls":
        tasks = [TaskType.MULTI_CLASS_CLS]
    elif task == "multi_label_cls":
        tasks = [TaskType.MULTI_LABEL_CLS]
    elif task == "classification":
        tasks = [TaskType.MULTI_CLASS_CLS, TaskType.MULTI_LABEL_CLS]
    elif task == "keypoint_detection":
        tasks = [TaskType.KEYPOINT_DETECTION]
    else:
        tasks = [TaskType(task.upper())]
    return tasks


def get_model_category_list(task: str) -> list[str]:
    """Return recipe paths for the speed / balance / accuracy models of the requested tasks.

    The canonical mapping lives in the application backend
    (``app.execution.common.geti_config_converter.TEMPLATE_ID_MAPPING``).
    Here we keep only the short list of category recipes so that library
    integration tests stay independent of the application package.

    Args:
        task: The task (or ``"all"``) for which to retrieve category recipes.

    Returns:
        A list of recipe path strings.
    """
    task_list = get_task_list(task.lower())
    recipes: list[str] = []

    for task_type in task_list:
        task_key = task_type.value.lower()
        recipes.extend(str(p) for p in CATEGORY_RECIPES_PER_TASK.get(task_key, []))

        # Classification category recipes are stored under multi_class_cls;
        # derive multi_label_cls variants when requested.
        if task_key == "multi_label_cls":
            recipes.extend(
                str(p).replace("multi_class_cls", task_key)
                for p in CATEGORY_RECIPES_PER_TASK.get("multi_class_cls", [])
            )

    return recipes


def pytest_configure(config):
    """Configure pytest options and set task, recipe, and recipe_ov lists.

    Args:
        config (pytest.Config): The pytest configuration object.

    Returns:
        None
    """
    task = config.getoption("--task")
    run_category_only = config.getoption("--run-category-only")

    # This assumes have getitune installed in environment.
    getitune_module = importlib.import_module("getitune")
    # Modify RECIPE_PATH based on the task
    recipe_path = Path(inspect.getfile(getitune_module)).parent / "recipe"
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

    # Run Model Category Recipes Only (i.e. model balance, accuracy, etc.)
    if run_category_only:
        target_recipe_list = get_model_category_list(task)

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
        "multi_class_cls": "tests/assets/classification_cifar10",
        "multi_class_cls_16bit": "tests/assets/classification_dataset_16bit",
        "multi_label_cls": "tests/assets/multilabel_classification_coco",
        "detection": "tests/assets/detection_coco",
        "instance_segmentation": "tests/assets/instance_segmentation_coco",
        "semantic_segmentation": "tests/assets/segmentation_pets",
        "keypoint_detection": "tests/assets/keypoint_detection_coco",
        "tiling_detection": "tests/assets/detection_coco",
    }
