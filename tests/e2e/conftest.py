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


@pytest.fixture(scope="session")
def fxt_ci_data_root() -> Path:
    data_root = Path(os.environ.get("CI_DATA_ROOT", "/home/validation/data"))
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
        return [OTXTaskType.ACTION_CLASSIFICATION]
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
        OTXTaskType.MULTI_CLASS_CLS: {
            "supervised": Path(fxt_ci_data_root / "v2/multiclass_classification/multiclass_CUB_small/1"),
            "unlabeled": Path(fxt_ci_data_root / "v2/multiclass_classification/semi-sl/CUB_unlabeled"),
        },
        OTXTaskType.MULTI_LABEL_CLS: Path(fxt_ci_data_root / "v2/multilabel_classification/multilabel_CUB_small/1"),
        OTXTaskType.H_LABEL_CLS: Path(fxt_ci_data_root / "v2/hlabel_classification/hlabel_CUB_small/1"),
        OTXTaskType.DETECTION: Path(fxt_ci_data_root / "v2/detection/bdd_small/1"),
        OTXTaskType.ROTATED_DETECTION: Path(fxt_ci_data_root / "v2/rotated_detection/subway"),
        OTXTaskType.INSTANCE_SEGMENTATION: {
            "non_tiling": Path(fxt_ci_data_root / "v2/instance_seg/wgisd_small/1"),
            "tiling": Path(fxt_ci_data_root / "v2/tiling_instance_seg/vitens_aeromonas_small/1"),
        },
        OTXTaskType.SEMANTIC_SEGMENTATION: {
            "supervised": Path(fxt_ci_data_root / "v2/semantic_seg/kvasir_small/1"),
            "unlabeled": Path(fxt_ci_data_root / "v2/semantic_seg/semi-sl/unlabeled_images/kvasir"),
        },
        OTXTaskType.ACTION_CLASSIFICATION: Path(
            fxt_ci_data_root / "v2/action/action_classification/ucf_kinetics_5percent_small",
        ),
        OTXTaskType.VISUAL_PROMPTING: Path(fxt_ci_data_root / "v2/visual_prompting/coco_car_person_medium"),
        OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING: Path(
            fxt_ci_data_root / "v2/zero_shot_visual_prompting/coco_car_person_medium",
        ),
        OTXTaskType.ANOMALY_CLASSIFICATION: Path(fxt_ci_data_root / "v2/anomaly/mvtec/hazelnut_large"),
        OTXTaskType.ANOMALY_DETECTION: Path(fxt_ci_data_root / "v2/anomaly/mvtec/hazelnut_large"),
        OTXTaskType.ANOMALY_SEGMENTATION: Path(fxt_ci_data_root / "v2/anomaly/mvtec/hazelnut_large"),
        OTXTaskType.KEYPOINT_DETECTION: Path(fxt_ci_data_root / "v2/keypoint_detection/coco_keypoint_medium"),
    }


@pytest.fixture()
def fxt_cli_override_command_per_task() -> dict:
    return {
        OTXTaskType.MULTI_CLASS_CLS: [],
        OTXTaskType.MULTI_LABEL_CLS: [],
        OTXTaskType.H_LABEL_CLS: [],
        OTXTaskType.DETECTION: [],
        OTXTaskType.ROTATED_DETECTION: [],
        OTXTaskType.INSTANCE_SEGMENTATION: [],
        OTXTaskType.SEMANTIC_SEGMENTATION: [],
        OTXTaskType.ACTION_CLASSIFICATION: [],
        OTXTaskType.VISUAL_PROMPTING: [],
        OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING: [],
        OTXTaskType.ANOMALY_CLASSIFICATION: [],
        OTXTaskType.ANOMALY_DETECTION: [],
        OTXTaskType.ANOMALY_SEGMENTATION: [],
        OTXTaskType.KEYPOINT_DETECTION: [],
    }
