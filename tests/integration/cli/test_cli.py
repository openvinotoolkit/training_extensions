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
RECIPE_LIST = [str(_.relative_to(RECIPE_PATH)) for _ in RECIPE_PATH.glob("**/*.yaml")]

# [TODO]: This is a temporary approach.
DATASET = {
    "classification": {
        "data_dir": "tests/assets/classification_dataset",
        "overrides": [
            "model.otx_model.config.head.num_classes=2",
        ],
    },
    "multilabel_classification": {
        "data_dir": "tests/assets/multilabel_classification",
        "overrides": [
            "model.otx_model.config.head.num_classes=2",
        ],
    },
    "detection": {
        "data_dir": "tests/assets/car_tree_bug",
        "overrides": ["model.otx_model.config.bbox_head.num_classes=3"],
    },
    "instance_segmentation": {
        "data_dir": "tests/assets/car_tree_bug",
        "overrides": [
            "model.otx_model.config.roi_head.bbox_head.num_classes=3",
            "model.otx_model.config.roi_head.mask_head.num_classes=3",
        ],
    },
    "segmentation": {
        "data_dir": "tests/assets/common_semantic_segmentation_dataset/supervised",
        "overrides": ["model.otx_model.config.decode_head.num_classes=2"],
    },
}


@pytest.mark.parametrize("recipe", RECIPE_LIST)
def test_otx_e2e(recipe: str, tmp_path: Path) -> None:
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
    task = recipe.split("/")[0]
    model_name = recipe.split("/")[1].split(".")[0]

    # 1) otx train
    tmp_path_train = tmp_path / f"otx_train_{model_name}"
    command_cfg = [
        "otx", "train",
        f"+recipe={recipe}",
        f"base.data_dir={DATASET[task]['data_dir']}",
        f"base.work_dir={tmp_path_train}",
        f"base.output_dir={tmp_path_train / 'outputs'}",
        "+debug=intg_test",
        *DATASET[task]['overrides'],
    ]

    with patch("sys.argv", command_cfg):
        main()

    # Currently, a simple output check
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
        "otx", "test",
        f"+recipe={recipe}",
        f"base.data_dir={DATASET[task]['data_dir']}",
        f"base.work_dir={tmp_path_test}",
        f"base.output_dir={tmp_path_test / 'outputs'}",
        *DATASET[task]['overrides'],
        f"checkpoint={ckpt_files[-1]}",
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert (tmp_path_test / "outputs").exists()
    assert (tmp_path_test / "outputs" / "otx_test.log").exists()
    assert (tmp_path_test / "outputs" / "lightning_logs").exists()
