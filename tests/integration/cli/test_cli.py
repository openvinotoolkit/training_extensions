# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
from otx.cli import main

# This assumes have OTX installed in environment.
otx_module = importlib.import_module("otx")
RECIPE_PATH = Path(inspect.getfile(otx_module)).parent / "recipe"
RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("**/*.yaml") if "_base_" not in p.parts]
RECIPE_OV_LIST = [str(p) for p in RECIPE_PATH.glob("**/openvino_model.yaml") if "_base_" not in p.parts]
RECIPE_LIST = set(RECIPE_LIST) - set(RECIPE_OV_LIST)


# [TODO]: This is a temporary approach.
@pytest.fixture()
def fxt_cli_override_command_per_task() -> dict:
    return {
        "multi_class_cls": ["--model.num_classes", "2"],
        "multi_label_cls": ["--model.num_classes", "2"],
        "h_label_cls": [
            "--model.num_classes",
            "7",
            "--model.num_multiclass_heads",
            "2",
            "--model.num_multilabel_classes",
            "3",
        ],
        "detection": ["--model.num_classes", "3"],
        "instance_segmentation": ["--model.num_classes", "3"],
        "semantic_segmentation": ["--model.num_classes", "2"],
        "action_classification": ["--model.num_classes", "2"],
        "action_detection": [
            "--model.num_classes",
            "5",
            "--model.topk",
            "3",
        ],
        "visual_prompting": [],
        "zero_shot_visual_prompting": ["--max_epochs", "1"],
    }


@pytest.mark.parametrize("recipe", RECIPE_LIST)
def test_otx_e2e(
    recipe: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
) -> None:
    """
    Test OTX CLI e2e commands.

    - 'otx train' with 2 epochs trainig
    - 'otx test' with output checkpoint from 'otx train'

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]
    if task in ("action_classification"):
        pytest.xfail(reason="xFail until this root cause is resolved on the Datumaro side.")

    # 1) otx train
    tmp_path_train = tmp_path / f"otx_train_{model_name}"
    command_cfg = [
        "otx",
        "train",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "2",
        *fxt_cli_override_command_per_task[task],
    ]

    with patch("sys.argv", command_cfg):
        main()

    # Currently, a simple output check
    assert (tmp_path_train / "outputs").exists()
    assert (tmp_path_train / "outputs" / "configs.yaml").exists()
    assert (tmp_path_train / "outputs" / "csv").exists()
    assert (tmp_path_train / "outputs" / "checkpoints").exists()
    ckpt_files = list((tmp_path_train / "outputs" / "checkpoints").glob(pattern="epoch_*.ckpt"))
    assert len(ckpt_files) > 0

    # 2) otx test
    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    command_cfg = [
        "otx",
        "test",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        fxt_accelerator,
        *fxt_cli_override_command_per_task[task],
        "--checkpoint",
        str(ckpt_files[-1]),
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert (tmp_path_test / "outputs").exists()
    assert (tmp_path_test / "outputs" / "csv").exists()


@pytest.mark.parametrize("recipe", RECIPE_LIST)
def test_otx_explain_e2e(
    recipe: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
) -> None:
    """
    Test OTX CLI explain e2e command.

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if ("_cls" not in task) and (task != "detection"):
        pytest.skip("Supported only for classification and detection task.")

    if "dino" in model_name:
        pytest.skip("Dino is not supported.")

    # otx explain
    tmp_path_explain = tmp_path / f"otx_explain_{model_name}"
    command_cfg = [
        "otx",
        "explain",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_explain / "outputs"),
        "--engine.device",
        fxt_accelerator,
        *fxt_cli_override_command_per_task[task],
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert (tmp_path_explain / "outputs").exists()
    assert (tmp_path_explain / "outputs" / "saliency_map.tiff").exists()


@pytest.mark.parametrize("recipe", RECIPE_OV_LIST)
def test_otx_ov_test(recipe: str, tmp_path: Path, fxt_target_dataset_per_task: dict) -> None:
    """
    Test OTX CLI e2e commands.

    - 'otx test' with OV model

    Args:
        recipe (str): The OV recipe to use for testing. (eg. 'classification/openvino_model.yaml')
        tmp_path (Path): The temporary path for storing the testing outputs.

    Returns:
        None
    """
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if task in ["multi_label_cls", "instance_segmentation", "h_label_cls"]:
        # OMZ doesn't have proper model for Pytorch MaskRCNN interface
        # TODO(Kirill):  Need to change this test when export enabled #noqa: TD003
        pytest.skip("OMZ doesn't have proper model for these types of tasks.")

    # otx test
    tmp_path_test = tmp_path / f"otx_test_{task}_{model_name}"
    command_cfg = [
        "otx",
        "test",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        "cpu",
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert (tmp_path_test / "outputs").exists()
    assert (tmp_path_test / "outputs" / "csv").exists()
    metric_result = list((tmp_path_test / "outputs" / "csv").glob(pattern="**/metrics.csv"))
    assert len(metric_result) > 0
