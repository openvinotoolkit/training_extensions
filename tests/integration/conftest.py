# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
from mmengine.config import Config as MMConfig
from otx.core.types.task import OTXTaskType


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--open-subprocess",
        action="store_true",
        help="Open subprocess for each CLI integration test case. "
        "This option can be used for easy memory management "
        "while running consecutive multiple tests (default: false).",
    )
    parser.addoption(
        "--task",
        action="store",
        default=None,
        type=OTXTaskType,
        help="Task type of OTX to use integration test.",
    )


@pytest.fixture(scope="module", autouse=True)
def fxt_open_subprocess(request: pytest.FixtureRequest) -> bool:
    """Open subprocess for each CLI integration test case.

    This option can be used for easy memory management
    while running consecutive multiple tests (default: false).
    """
    return request.config.getoption("--open-subprocess")


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
    if task is not None:
        recipe_path = find_recipe_folder(recipe_path, task.value.lower())
        task_list = [task]
    else:
        task_list = [task_type for task_type in OTXTaskType if task_type != OTXTaskType.DETECTION_SEMI_SL]

    # Update RECIPE_LIST
    recipe_list = [str(p) for p in recipe_path.glob("**/*.yaml") if "_base_" not in p.parts]
    recipe_ov_list = [str(p) for p in recipe_path.glob("**/openvino_model.yaml") if "_base_" not in p.parts]
    recipe_list = set(recipe_list) - set(recipe_ov_list)

    config.TASK_LIST = task_list
    config.RECIPE_LIST = recipe_list
    config.RECIPE_OV_LIST = recipe_ov_list


def pytest_generate_tests(metafunc):
    """Generate test cases for pytest based on the provided fixtures.

    This is to ensure that they behave separately per task.

    Args:
        metafunc: The metafunc object containing information about the test function.

    Returns:
        None
    """
    if "task" in metafunc.fixturenames:
        metafunc.parametrize("task", metafunc.config.TASK_LIST, scope="session")
    if "recipe" in metafunc.fixturenames:
        metafunc.parametrize(
            "recipe",
            metafunc.config.RECIPE_LIST,
            scope="session",
            ids=lambda x: "/".join(Path(x).parts[-2:]),
        )
    if "ov_recipe" in metafunc.fixturenames:
        if metafunc.config.RECIPE_OV_LIST:
            metafunc.parametrize(
                "ov_recipe",
                metafunc.config.RECIPE_OV_LIST,
                scope="session",
                ids=lambda x: "/".join(Path(x).parts[-2:]),
            )
        else:
            pytest.skip("No OpenVINO recipe found for the task.")


@pytest.fixture(scope="session")
def fxt_asset_dir() -> Path:
    return Path(__file__).parent.parent / "assets"


@pytest.fixture(scope="session")
def fxt_rtmdet_tiny_config(fxt_asset_dir: Path) -> MMConfig:
    config_path = fxt_asset_dir / "mmdet_configs" / "rtmdet_tiny_8xb32-300e_coco.py"

    return MMConfig.fromfile(config_path)


# [TODO]: This is a temporary approach.
@pytest.fixture()
def fxt_target_dataset_per_task() -> dict:
    return {
        "multi_class_cls": "tests/assets/classification_dataset",
        "multi_label_cls": "tests/assets/multilabel_classification",
        "h_label_cls": "tests/assets/hlabel_classification",
        "detection": "tests/assets/car_tree_bug",
        "rotated_detection": "tests/assets/car_tree_bug",
        "instance_segmentation": "tests/assets/car_tree_bug",
        "semantic_segmentation": "tests/assets/common_semantic_segmentation_dataset/supervised",
        "action_classification": "tests/assets/action_classification_dataset/",
        "action_detection": "tests/assets/action_detection_dataset/",
        "visual_prompting": "tests/assets/car_tree_bug",
        "zero_shot_visual_prompting": "tests/assets/car_tree_bug",
    }


@pytest.fixture()
def fxt_cli_override_command_per_task() -> dict:
    return {
        "multi_class_cls": [],
        "multi_label_cls": [],
        "h_label_cls": [
            "--model.num_multiclass_heads",
            "2",
            "--model.num_multilabel_classes",
            "3",
        ],
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
        "zero_shot_visual_prompting": [
            "--max_epochs",
            "1",
            "--disable-infer-num-classes",
        ],
        "tile": [
            "--data.config.tile_config.grid_size",
            "[1,1]",
        ],
    }
