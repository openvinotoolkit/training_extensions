# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import inspect
import logging
import re
from pathlib import Path

import pytest
from unittest.mock import patch

import pytest
from otx.cli import main

log = logging.getLogger(__name__)

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
}


def _check_relative_metric_diff(ref: float, value: float, eps: float) -> None:
    assert ref >= 0
    assert value >= 0
    assert eps >= 0

    avg = max(0.5 * (ref + value), 1e-9)
    diff = abs(value - ref)

    assert diff / avg <= eps, f"Relative difference exceeded {eps} threshold. Absolute difference: {diff}"


@pytest.fixture(scope="module", autouse=True)
def fxt_local_seed() -> int:
    """The number of repetition for each test case.

    The random seed will be set for [0, fxt_num_repeat - 1]. Default is one.
    """
    selected_seed = 7
    msg = f"seed : {selected_seed}"
    log.info(msg)
    return selected_seed


TASK_NAME_TO_MAIN_METRIC_NAME = {
    "semantic_segmentation": "test/mIoU",
    "multi_label_cls": "test/accuracy",
    "multi_class_cls": "test/accuracy",
}


@pytest.mark.parametrize("recipe", RECIPE_LIST)
def test_otx_export_infer(recipe: str, tmp_path: Path, fxt_local_seed: int, fxt_accelerator: str, capfd: "pytest.CaptureFixture") -> None:
    """
    Test OTX CLI e2e commands.

    - 'otx train' with 2 epochs training
    - 'otx test' with output checkpoint from 'otx train'
    - 'otx export' with output checkpoint from 'otx train'
    - 'otx test' with the exported model
    - compare accuracy of the exported model vs the original accuracy

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    task = recipe.split("/")[-2]

    if not task in TASK_NAME_TO_MAIN_METRIC_NAME or "dino_v2" in recipe:
        pytest.skip(f"Inference pipeline for {recipe} is not implemented")

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
        "--seed",
        f"{fxt_local_seed}",
        *DATASET[task]["overrides"],
    ]

    with patch("sys.argv", command_cfg):
        main()

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

    # 3) otx export
    format_to_ext = {"OPENVINO": "xml"}  # [TODO](@Vlad): extend to "ONNX": "onnx"

    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    for fmt in format_to_ext:
        command_cfg = [
            "otx",
            "export",
            "--config",
            recipe,
            "--data_root",
            DATASET[task]["data_root"],
            "--engine.work_dir",
            str(tmp_path_test / "outputs"),
            *DATASET[task]["overrides"],
            "--checkpoint",
            str(ckpt_files[-1]),
            "--export_config.export_format",
            f"{fmt}",
        ]

        with patch("sys.argv", command_cfg):
            main()

        assert (tmp_path_test / "outputs").exists()
        assert (tmp_path_test / "outputs" / f"exported_model.{format_to_ext[fmt]}").exists()


    # 4) infer of the exported models
    task = recipe.split("/")[-2]
    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    if "_cls" in recipe:
        export_test_recipe = f"src/otx/recipe/classification/{task}/openvino_model.yaml"
    else:
        export_test_recipe = f"src/otx/recipe/{task}/openvino_model.yaml"
    exported_model_path = str(tmp_path_test / "outputs" / "exported_model.xml")

    command_cfg = [
        "otx",
        "test",
        "--config",
        export_test_recipe,
        "--data_root",
        DATASET[task]["data_root"],
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        "cpu",
        *DATASET[task]["overrides"],
        "--model.model_name",
        exported_model_path,
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert (tmp_path_test / "outputs").exists()

    out, _ = capfd.readouterr()
    assert TASK_NAME_TO_MAIN_METRIC_NAME[task] in out
    torch_acc, ov_acc = tuple(re.findall(f"{TASK_NAME_TO_MAIN_METRIC_NAME[task]}\s*│\s*(\d+[.]\d+)", out))
    torch_acc, ov_acc = float(torch_acc), float(ov_acc)

    msg = f"Recipe: {recipe}, (torch_accuracy, ov_accuracy): {torch_acc} , {ov_acc}"
    log.info(msg)

    _check_relative_metric_diff(torch_acc, ov_acc, 0.1)
