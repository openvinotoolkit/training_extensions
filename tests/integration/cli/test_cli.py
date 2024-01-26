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
DATASET = {
    "multi_class_cls": {
        "data_root": "tests/assets/classification_dataset",
        "overrides": ["--model.num_classes", "2"],
    },
    "multi_label_cls": {
        "data_root": "tests/assets/multilabel_classification",
        "overrides": ["--model.num_classes", "2"],
    },
    "h_label_cls": {
        "data_root": "tests/assets/hlabel_classification",
        "overrides": [
            "--model.num_classes",
            "7",
            "--model.num_multiclass_heads",
            "2",
            "--model.num_multilabel_classes",
            "3",
        ],
    },
    "detection": {
        "data_root": "tests/assets/car_tree_bug",
        "overrides": ["--model.num_classes", "3"],
    },
    "instance_segmentation": {
        "data_root": "tests/assets/car_tree_bug",
        "overrides": ["--model.num_classes", "3"],
    },
    "semantic_segmentation": {
        "data_root": "tests/assets/common_semantic_segmentation_dataset/supervised",
        "overrides": ["--model.num_classes", "2"],
    },
    "action_classification": {
        "data_root": "tests/assets/action_classification_dataset/",
        "overrides": ["--model.num_classes", "2"],
    },
    "action_detection": {
        "data_root": "tests/assets/action_detection_dataset/",
        "overrides": [
            "--model.num_classes",
            "5",
            "--model.topk",
            "3",
        ],
    },
    "visual_prompting": {
        "data_root": "tests/assets/car_tree_bug",
        "overrides": [],
    },
    "zero_shot_visual_prompting": {
        "data_root": "tests/assets/car_tree_bug",
        "overrides": [],
    },
}


@pytest.mark.parametrize("recipe", RECIPE_LIST)
def test_otx_e2e(recipe: str, tmp_path: Path, fxt_accelerator: str) -> None:
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

    # 1) otx train
    tmp_path_train = tmp_path / f"otx_train_{model_name}"
    command_cfg = [
        "otx",
        "train",
        "--config",
        recipe,
        "--data_root",
        DATASET[task]["data_root"],
        "--engine.work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "2",
        *DATASET[task]["overrides"],
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
        DATASET[task]["data_root"],
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        fxt_accelerator,
        *DATASET[task]["overrides"],
        "--checkpoint",
        str(ckpt_files[-1]),
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert (tmp_path_test / "outputs").exists()
    assert (tmp_path_test / "outputs" / "csv").exists()


@pytest.mark.parametrize("recipe", RECIPE_OV_LIST)
def test_otx_ov_test(recipe: str, tmp_path: Path) -> None:
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

    if task == "instance_segmentation":
        # OMZ doesn't have proper model for Pytorch MaskRCNN interface
        # TODO(Kirill):  Need to change this test when export enabled #noqa: TD003
        pytest.skip("OMZ doesn't have proper model for Pytorch MaskRCNN interface.")

    # otx test
    tmp_path_test = tmp_path / f"otx_test_{task}_{model_name}"
    command_cfg = [
        "otx",
        "test",
        "--config",
        recipe,
        "--data_root",
        DATASET[task]["data_root"],
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
