"""Tests the methods in config."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.config.visual_prompting_config import (
    get_configurable_parameters,
    get_visual_promtping_config,
    update_visual_prompting_config,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_visual_promtping_config(tmpdir, mocker):
    """Test get_visual_promtping_config."""
    task_name = "sam_vit_b"
    mocker_otx_config = mocker.patch("otx.api.configuration.configurable_parameters.ConfigurableParameters")
    output_path = str(tmpdir.mkdir("visual_prompting_training_test"))
    config = get_visual_promtping_config(task_name, mocker_otx_config, output_path)

    assert isinstance(config, DictConfig)
    assert config.get("dataset", False)
    assert config.get("model", False)
    assert config.get("optimizer", False)
    assert config.get("callback", False)
    assert config.get("trainer", False)


@e2e_pytest_unit
def test_get_visual_promtping_config_with_already_saved_config(tmpdir, mocker):
    """Test get_visual_promtping_config with already saved config."""
    mocker_conf_load = mocker.patch("omegaconf.OmegaConf.load", return_value="foo")
    mocker_conf_save = mocker.patch("omegaconf.OmegaConf.to_yaml")
    mocker_path_write_text = mocker.patch("pathlib.Path.write_text")

    task_name = "sam_vit_b"
    otx_config = mocker.patch("otx.api.configuration.configurable_parameters.ConfigurableParameters")
    output_path = str(tmpdir.mkdir("visual_prompting_training_test"))
    config = get_visual_promtping_config(task_name, otx_config, output_path)

    assert config == "foo"
    mocker_conf_load.assert_called_once()
    mocker_conf_save.assert_called_once()
    mocker_path_write_text.assert_called_once()


@e2e_pytest_unit
@pytest.mark.parametrize("weight_file", [None, "weight.file"])
def test_get_configurable_parameters(tmpdir, weight_file: str):
    """Test get_configurable_parameters."""
    output_path = Path(tmpdir.mkdir("visual_prompting_training_test"))
    model_name = "sam_vit_b"
    config_path = Path(f"otx/algorithms/visual_prompting/configs/{model_name}/config.yaml")
    config = get_configurable_parameters(
        output_path=output_path,
        model_name=model_name,
        config_path=config_path,
        weight_file=weight_file
    )

    assert isinstance(config, DictConfig)
    assert config.get("dataset", False)
    assert config.get("model", False)
    assert config.get("optimizer", False)
    assert config.get("callback", False)
    assert config.get("trainer", False)
    if weight_file is not None:
        assert config.trainer.resume_from_checkpoint == weight_file


@e2e_pytest_unit
def test_get_configurable_parameters_without_any_arguments(tmpdir):
    """Test get_configurable_parameters without any arguments."""
    output_path = Path(tmpdir.mkdir("visual_prompting_training_test"))
    with pytest.raises(ValueError):
        get_configurable_parameters(output_path)


@e2e_pytest_unit
def test_update_visual_prompting_config():
    """Test update_visual_prompting_config."""
    otx_config = OmegaConf.create({
        "groups": ["learning_parameters"],
        "learning_parameters": {
            "parameters": ["param1"],
            "param1": "updated_value1"
        },
        "parameters": []
    })
    visual_prompting_config = OmegaConf.create({"param1": "value1", "param2": "value2"})

    update_visual_prompting_config(visual_prompting_config, otx_config)
    
    assert visual_prompting_config["param1"] == "updated_value1"
