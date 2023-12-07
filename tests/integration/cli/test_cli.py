# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra.core.global_hydra import GlobalHydra
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
            "model.otx_model.config.data_preprocessor.num_classes=2",
        ],
    },
    "detection": {
        "data_dir": "tests/assets/car_tree_bug",
        "overrides": ["model.otx_model.config.bbox_head.num_classes=3"],
    },
    "segmentation": {
        "data_dir": "tests/assets/common_semantic_segmentation_dataset/supervised",
        "overrides": ["model.otx_model.config.decode_head.num_classes=2"],
    },
}


@pytest.mark.parametrize("recipe", RECIPE_LIST)
def test_otx_train(recipe: str, tmp_path: Path) -> None:
    """
    Test the 'otx train' command with 1 epochs trainig and check outputs.

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    task = recipe.split("/")[0]
    model_name = recipe.split("/")[1].split(".")[0]
    tmp_path = tmp_path / f"otx_train_{model_name}"
    command_cfg = [
        "otx", "train",
        f"+recipe={recipe}",
        f"base.data_dir={DATASET[task]['data_dir']}",
        f"base.work_dir={tmp_path}",
        f"base.output_dir={tmp_path / 'outputs'}",
        "+debug=intg_test",
        *DATASET[task]['overrides'],
    ]

    with patch("sys.argv", command_cfg):
        main()

    # Clear the Hydra instances.
    GlobalHydra().clear()

    # Currently, a simple output check
    assert (tmp_path / "outputs").exists()
    assert (tmp_path / "outputs" / "otx_train.log").exists()
    assert (tmp_path / "outputs" / "csv").exists()
    assert (tmp_path / "outputs" / "csv" / "version_0").exists()
    assert (tmp_path / "outputs" / "csv" / "version_0" / "checkpoints").exists()
    ckpt_files = list((tmp_path / "outputs" / "csv" / "version_0" / "checkpoints").glob(pattern="*.ckpt"))
    assert len(ckpt_files) > 0
