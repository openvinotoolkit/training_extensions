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
RECIPE_LIST = [str(_) for _ in RECIPE_PATH.glob("**/*.yaml")]

# [TODO]: This is a temporary approach.
DATASET = {
    "classification": {
        "data_dir": "tests/assets/classification_dataset",
        "overrides": [
            "--model.otx_model.config.head.num_classes",
            "2",
        ],
    },
    "detection": {
        "data_dir": "tests/assets/car_tree_bug",
        "overrides": [
            "--model.otx_model.config.bbox_head.num_classes",
            "3",
        ],
    },
    "instance_segmentation": {
        "data_dir": "tests/assets/car_tree_bug",
        "overrides": [
            "--model.otx_model.config.roi_head.bbox_head.num_classes",
            "3",
            "--model.otx_model.config.roi_head.mask_head.num_classes",
            "3",
        ],
    },
    "segmentation": {
        "data_dir": "tests/assets/common_semantic_segmentation_dataset/supervised",
        "overrides": [
            "--model.otx_model.config.decode_head.num_classes",
            "2",
        ],
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
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    # 1) otx train
    tmp_path_train = tmp_path / f"otx_train_{model_name}"
    command_cfg = [
        "otx",
        "train",
        "--config",
        str(recipe),
        "--engine.work_dir",
        str(tmp_path_train),
        "--data.config.data_root",
        DATASET[task]["data_dir"],
        "--max_epochs",
        "2",
        *DATASET[task]["overrides"],
    ]

    with patch("sys.argv", command_cfg):
        main()

    # Currently, a simple output check
    assert tmp_path_train.exists()
    assert (tmp_path_train / "configs.yaml").exists()
    assert (tmp_path_train / "lightning_logs").exists()
    assert (tmp_path_train / "lightning_logs" / "version_0").exists()
    assert (tmp_path_train / "lightning_logs" / "version_0" / "checkpoints").exists()
    ckpt_files = list((tmp_path_train / "lightning_logs" / "version_0" / "checkpoints").glob(pattern="epoch_*.ckpt"))
    assert len(ckpt_files) > 0

    # 2) otx test
    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    command_cfg = [
        "otx",
        "test",
        "--config",
        str(recipe),
        "--engine.work_dir",
        str(tmp_path_test),
        "--data.config.data_root",
        DATASET[task]["data_dir"],
        *DATASET[task]["overrides"],
        f"--checkpoint={ckpt_files[-1]}",
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert tmp_path_test.exists()
    assert (tmp_path_test / "configs.yaml").exists()
    assert (tmp_path_test / "lightning_logs").exists()
