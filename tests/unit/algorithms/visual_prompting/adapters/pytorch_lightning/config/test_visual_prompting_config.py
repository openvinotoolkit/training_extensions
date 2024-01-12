"""Tests the methods in config."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional

import pytest
from omegaconf import DictConfig, OmegaConf

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.config.visual_prompting_config import (
    get_visual_promtping_config,
    update_visual_prompting_config,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
@pytest.mark.parametrize(
    "model_checkpoint,resume_from_checkpoint",
    [
        (None, None),
        ("model_checkpoint.ckpt", None),
        (None, "resume_from_checkpoint.ckpt"),
    ],
)
@pytest.mark.parametrize("mode", ["train", "inference"])
def test_get_visual_promtping_config(
    tmpdir, mocker, mode: str, model_checkpoint: Optional[str], resume_from_checkpoint: Optional[str]
):
    """Test get_visual_promtping_config."""
    task_name = "sam_vit_b"
    mocker_otx_config = mocker.patch("otx.api.configuration.configurable_parameters.ConfigurableParameters")
    config_dir = str(tmpdir.mkdir("visual_prompting_training_test"))
    config = get_visual_promtping_config(
        task_name=task_name,
        otx_config=mocker_otx_config,
        config_dir=config_dir,
        mode=mode,
        model_checkpoint=model_checkpoint,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    assert isinstance(config, DictConfig)
    assert config.get("dataset", False)
    assert config.get("model", False)
    assert config.get("optimizer", False)
    assert config.get("callback", False)
    assert config.get("trainer", False)
    if mode == "train":
        if model_checkpoint:
            assert config.get("model").get("checkpoint", None) == model_checkpoint
        else:
            assert config.get("model").get("checkpoint", None) != model_checkpoint
        assert config.get("trainer").get("resume_from_checkpoint", None) == resume_from_checkpoint


@e2e_pytest_unit
def test_update_visual_prompting_config():
    """Test update_visual_prompting_config."""
    otx_config = OmegaConf.create(
        {
            "groups": ["learning_parameters", "pot_parameters", "postprocessing", "algo_backend"],
            "learning_parameters": {"parameters": ["param1"], "param1": "updated_value1"},
            "pot_parameters": {"parameters": ["param2"], "param2": "updated_value2"},
            "postprocessing": {"parameters": ["param3"], "param3": "updated_value3"},
            "algo_backend": {"parameters": ["param4"], "param4": "updated_value4"},
            "parameters": [],
        }
    )
    visual_prompting_config = OmegaConf.create(
        {"param1": "value1", "param2": "value2", "param3": "value3", "param4": "value4", "param5": "value5"}
    )

    update_visual_prompting_config(visual_prompting_config, otx_config)

    assert visual_prompting_config["param1"] == "updated_value1"
    assert visual_prompting_config["param2"] == "updated_value2"
    assert visual_prompting_config["param3"] == "updated_value3"
    assert visual_prompting_config["param4"] == "updated_value4"
    assert visual_prompting_config["param5"] == "value5"
