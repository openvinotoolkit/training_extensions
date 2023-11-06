# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.adapters.torch.mmengine.utils.runner_config import (
    update_train_config,
)
from pytest_mock.plugin import MockerFixture


@pytest.fixture()
def func_args() -> dict:
    return {}


@pytest.fixture()
def config() -> dict:
    return {"max_epochs": 10, "val_interval": 2}


def test_update_train_config_with_max_epochs(func_args: dict, mocker: MockerFixture) -> None:
    mocker.patch("otx.v2.adapters.torch.mmengine.utils.runner_config.get_device", return_value="cuda")
    precision = "float32"
    updated_config = Config({"test": "test1"})
    default_config = Config({"max_epochs": 10, "val_interval": 2, "precision": "float32"})
    mock_dataloader = mocker.MagicMock()
    update_train_config(
        train_dataloader=mock_dataloader,
        arguments=func_args,
        default_config=default_config,
        config=updated_config,
    )

    assert updated_config["train_cfg"]["by_epoch"] is True
    assert updated_config["train_cfg"]["max_epochs"] == default_config["max_epochs"]
    assert updated_config["optim_wrapper"]["type"] == "AmpOptimWrapper"
    assert updated_config["optim_wrapper"]["dtype"] == precision


def test_update_train_config_with_max_iters(func_args: dict, config: dict, mocker: MockerFixture) -> None:
    mocker.patch("otx.v2.adapters.torch.mmengine.utils.runner_config.get_device", return_value="cuda")
    func_args["max_iters"] = 100
    func_args["max_epochs"] = 100
    func_args["precision"] = "float16"
    updated_config = Config({"test": "test1"})
    mock_dataloader = mocker.MagicMock()
    with pytest.raises(ValueError, match="Only one of `max_epochs` or `max_iters`"):
        update_train_config(
            train_dataloader=mock_dataloader,
            arguments=func_args, default_config=config,
            config=updated_config,
        )
    config["max_epochs"] = None
    func_args["max_epochs"] = None
    update_train_config(
        train_dataloader=mock_dataloader,
        arguments=func_args, default_config=config, config=updated_config,
    )

    assert updated_config["train_cfg"]["by_epoch"] is False
    assert updated_config["train_cfg"]["max_iters"] == func_args["max_iters"]
    assert updated_config["optim_wrapper"]["type"] == "AmpOptimWrapper"
    assert updated_config["optim_wrapper"]["dtype"] == func_args["precision"]


def test_update_train_config_raises_value_error(func_args: dict, config: dict, mocker: MockerFixture) -> None:
    config["max_iters"] = 100
    config["max_epochs"] = 10

    mock_dataloader = mocker.MagicMock()
    result_config = Config({})
    with pytest.raises(ValueError, match="Only one of `max_epochs` or `max_iters`"):
        update_train_config(
            train_dataloader=mock_dataloader,
            arguments=func_args,
            default_config=config,
            config=result_config,
        )


def test_update_train_config_with_train_cfg_in_kwargs(func_args: dict, mocker: MockerFixture) -> None:
    default_config = Config({"val_interval": 3})
    updated_config = Config({})
    mock_dataloader = mocker.MagicMock()
    update_train_config(
        train_dataloader=mock_dataloader,
        arguments=func_args, default_config=default_config, config=updated_config,
    )

    assert updated_config["train_cfg"]["val_interval"] == default_config["val_interval"]
