# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.v2.adapters.torch.mmengine.utils.runner_config import (
    update_train_config,
    update_val_test_config,
)
from pytest_mock.plugin import MockerFixture


@pytest.fixture()
def func_args() -> dict:
    return {}


@pytest.fixture()
def config() -> dict:
    return {"max_epochs": 10, "val_interval": 2}


@pytest.fixture()
def kwargs(mocker: MockerFixture) -> dict:
    return {"train_dataloader": mocker.MagicMock()}


def test_update_train_config_with_max_epochs(func_args: dict, config: dict, kwargs: dict, mocker: MockerFixture) -> None:
    mocker.patch("otx.v2.adapters.torch.mmengine.utils.runner_config.get_device", return_value="cuda")
    precision = "float32"
    updated_config, updated_kwargs = update_train_config(func_args, config, precision, **kwargs)

    assert updated_config == config
    assert updated_kwargs["train_cfg"]["by_epoch"] is True
    assert updated_kwargs["train_cfg"]["max_epochs"] == config["max_epochs"]
    assert updated_kwargs["optim_wrapper"]["type"] == "AmpOptimWrapper"
    assert updated_kwargs["optim_wrapper"]["dtype"] == precision


def test_update_train_config_with_max_iters(func_args: dict, config: dict, kwargs: dict, mocker: MockerFixture) -> None:
    mocker.patch("otx.v2.adapters.torch.mmengine.utils.runner_config.get_device", return_value="cuda")
    func_args["max_iters"] = 100
    precision = "float16"
    with pytest.raises(ValueError, match="Only one of `max_epochs` or `max_iters`"):
        update_train_config(func_args, config, precision, **kwargs)
    config["max_epochs"] = None
    updated_config, updated_kwargs = update_train_config(func_args, config, precision, **kwargs)

    assert updated_config == config
    assert updated_kwargs["train_cfg"]["by_epoch"] is False
    assert updated_kwargs["train_cfg"]["max_iters"] == func_args["max_iters"]
    assert updated_kwargs["optim_wrapper"]["type"] == "AmpOptimWrapper"
    assert updated_kwargs["optim_wrapper"]["dtype"] == precision


def test_update_train_config_raises_value_error(func_args: dict, config: dict, kwargs: dict) -> None:
    func_args["max_iters"] = 100
    config["max_epochs"] = 10

    with pytest.raises(ValueError, match="Only one of `max_epochs` or `max_iters`"):
        update_train_config(func_args, config, None, **kwargs)


def test_update_train_config_with_train_cfg_in_kwargs(func_args: dict, config: dict) -> None:
    kwargs = {"train_cfg": {"val_interval": 3}}
    updated_config, updated_kwargs = update_train_config(func_args, config, None, **kwargs)

    assert updated_config == config
    assert updated_kwargs["train_cfg"]["val_interval"] == kwargs["train_cfg"]["val_interval"]

def test_update_train_config_with_train_dataloader_as_dict_in_config(func_args: dict) -> None:
    config = {"train_dataloader": {"type": "DatasetEntity"}}
    kwargs = {}
    updated_config, updated_kwargs = update_train_config(func_args, config, None, **kwargs)

    assert updated_config["train_dataloader"] is None
    assert updated_config["train_cfg"] is None
    assert updated_config["optim_wrapper"] is None
    assert updated_kwargs == {}


def test_update_val_test_config(mocker: MockerFixture) -> None:
    mocker.patch("otx.v2.adapters.torch.mmengine.utils.runner_config.get_value_from_config", return_value=None)
    precision = "fp16"
    mock_kwargs = {"val_dataloader": mocker.MagicMock(), "test_dataloader": mocker.MagicMock()}
    updated_config, updated_kwargs = update_val_test_config({}, {}, precision, **mock_kwargs)

    assert updated_config == {}
    assert "val_dataloader" in updated_kwargs
    assert "val_cfg" in updated_kwargs
    assert "val_evaluator" in updated_kwargs
    assert updated_kwargs["val_cfg"]["fp16"]
    assert "test_dataloader" in updated_kwargs
    assert "test_cfg" in updated_kwargs
    assert "test_evaluator" in updated_kwargs
    assert updated_kwargs["val_cfg"]["fp16"]

    mock_kwargs = {}
    mock_config = {"val_dataloader": {}, "test_dataloader": {}}
    updated_config, updated_kwargs = update_val_test_config({}, mock_config, precision, **mock_kwargs)

    assert updated_kwargs == {}
    assert "val_dataloader" in updated_config
    assert "val_cfg" in updated_config
    assert "val_evaluator" in updated_config
    assert updated_config["val_dataloader"] is None
    assert updated_config["val_cfg"] is None
    assert updated_config["val_evaluator"] is None
    assert "test_dataloader" in updated_config
    assert "test_cfg" in updated_config
    assert "test_evaluator" in updated_config
    assert updated_config["test_dataloader"] is None
    assert updated_config["test_cfg"] is None
    assert updated_config["test_evaluator"] is None
