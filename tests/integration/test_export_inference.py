# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import inspect
import logging
from pathlib import Path

import pytest
from otx.cli.export import otx_export
from otx.cli.test import otx_test
from otx.cli.train import otx_train

log = logging.getLogger(__name__)

# This assumes have OTX installed in environment.
otx_module = importlib.import_module("otx")
RECIPE_PATH = Path(inspect.getfile(otx_module)).parent / "recipe"
ALL_RECIPE_LIST = [str(_.relative_to(RECIPE_PATH)) for _ in RECIPE_PATH.glob("**/*.yaml")]
RECIPE_OV_LIST = [str(_.relative_to(RECIPE_PATH)) for _ in RECIPE_PATH.glob("**/openvino_model.yaml")]
RECIPE_LIST = set(ALL_RECIPE_LIST) - set(RECIPE_OV_LIST)

# [TODO]: This is a temporary approach.
DATASET = {
    "multiclass_classification": {
        "data_dir": "tests/assets/classification_dataset",
        "overrides": [
            "model.otx_model.num_classes=2",
        ],
    },
    "multilabel_classification": {
        "data_dir": "tests/assets/multilabel_classification",
        "overrides": [
            "model.otx_model.num_classes=2",
        ],
    },
    "hlabel_classification": {
        "data_dir": "tests/assets/hlabel_classification",
        "overrides": [
            "model.otx_model.num_classes=7",
            "model.otx_model.num_multiclass_heads=2",
            "model.otx_model.num_multilabel_classes=3",
        ],
    },
    "detection": {
        "data_dir": "tests/assets/car_tree_bug",
        "overrides": ["model.otx_model.num_classes=3"],
    },
    "instance_segmentation": {
        "data_dir": "tests/assets/car_tree_bug",
        "overrides": [
            "model.otx_model.num_classes=3",
        ],
    },
    "segmentation": {
        "data_dir": "tests/assets/common_semantic_segmentation_dataset/supervised",
        "overrides": ["model.otx_model.num_classes=2"],
    },
    "action_classification": {
        "data_dir": "tests/assets/action_classification_dataset/",
        "overrides": ["model.otx_model.num_classes=2"],
    },
    "action_detection": {
        "data_dir": "tests/assets/action_detection_dataset/",
        "overrides": [
            "model.otx_model.num_classes=5",
            "model.otx_model.topk=3",
        ],
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
    "segmentation": "test/mIoU",
    "multilabel_classification": "test/accuracy",
    "multiclass_classification": "test/accuracy",
    "detection" : "test/map_50",
    "instance_segmentation" : "test/map_50",
}


@pytest.mark.parametrize("recipe", RECIPE_LIST)
def test_otx_e2e(recipe: str, tmp_path: Path, fxt_local_seed: int, fxt_accelerator: str) -> None:
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
    if any(
        task_name in recipe
        for task_name in [
            "hlabel_classification",
            "dino_v2",
            "instance_segmentation",
            "action_classification",
            "action_detection",
            "visual_prompting",
        ]
    ):
        pytest.skip(f"Inference pipeline for {recipe} is not implemented")

    task = recipe.split("/")[0]
    model_name = recipe.split("/")[1].split(".")[0]

    # 1) otx train
    tmp_path_train = tmp_path / f"otx_train_{model_name}"
    command_cfg = [
        f"+recipe={recipe}",
        f"base.data_dir={DATASET[task]['data_dir']}",
        f"base.work_dir={tmp_path_train}",
        f"base.output_dir={tmp_path_train / 'outputs'}",
        "+debug=intg_test",
        f"seed={fxt_local_seed}",
        *DATASET[task]["overrides"],
    ]
    otx_train(command_cfg)

    assert (tmp_path_train / "outputs").exists()
    assert (tmp_path_train / "outputs" / "otx_train.log").exists()
    assert (tmp_path_train / "outputs" / "csv").exists()
    assert (tmp_path_train / "outputs").exists()
    assert (tmp_path_train / "outputs" / "checkpoints").exists()

    ckpt_files = list((tmp_path_train / "outputs" / "checkpoints").glob(pattern="epoch_*.ckpt"))
    assert len(ckpt_files) > 0

    # 2) otx test
    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    command_cfg = [
        f"+recipe={recipe}",
        f"base.data_dir={DATASET[task]['data_dir']}",
        f"base.work_dir={tmp_path_test}",
        f"base.output_dir={tmp_path_test / 'outputs'}",
        f"seed={fxt_local_seed}",
        f"trainer={fxt_accelerator}",
        *DATASET[task]["overrides"],
        f"checkpoint={ckpt_files[-1]}",
    ]

    torch_acc = otx_test(command_cfg)[TASK_NAME_TO_MAIN_METRIC_NAME[task]]

    assert (tmp_path_test / "outputs").exists()
    assert (tmp_path_test / "outputs" / "otx_test.log").exists()
    assert (tmp_path_test / "outputs" / "lightning_logs").exists()

    # 3) otx export
    format_to_ext = {"OPENVINO": "xml"}  # [TODO](@Vlad): extend to "ONNX": "onnx"

    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    for fmt in format_to_ext:
        command_cfg = [
            f"+recipe={recipe}",
            f"base.data_dir={DATASET[task]['data_dir']}",
            f"base.work_dir={tmp_path_test}",
            f"base.output_dir={tmp_path_test / 'outputs'}",
            *DATASET[task]["overrides"],
            f"checkpoint={ckpt_files[-1]}",
            f"model.export_config.export_format={fmt}",
        ]

        otx_export(command_cfg)

        assert (tmp_path_test / "outputs").exists()
        assert (tmp_path_test / "outputs" / f"exported_model.{format_to_ext[fmt]}").exists()

    # 4) infer of the exported models
    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    task = recipe.split("/")[0]
    export_test_recipe = f"{task}/openvino_model.yaml"
    exported_model_path = str(tmp_path_test / "outputs" / "exported_model.xml")

    command_cfg = [
        f"+recipe={export_test_recipe}",
        f"base.data_dir={DATASET[task]['data_dir']}",
        f"base.work_dir={tmp_path_test}",
        f"base.output_dir={tmp_path_test / 'outputs'}",
        *DATASET[task]["overrides"],
        f"model.otx_model.config.model_name={exported_model_path}",
    ]

    ov_acc = otx_test(command_cfg)[TASK_NAME_TO_MAIN_METRIC_NAME[task]]

    msg = f"Recipe: {recipe}, (torch_accuracy, ov_accuracy): {torch_acc} , {ov_acc}"
    log.info(msg)

    _check_relative_metric_diff(torch_acc, ov_acc, 0.1)
