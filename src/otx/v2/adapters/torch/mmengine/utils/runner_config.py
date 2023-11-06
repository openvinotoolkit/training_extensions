"""OTX adapters.torch.mmengine functions related to Runner config."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from mmengine.device import get_device

from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config

if TYPE_CHECKING:
    import torch


def get_value_from_config(
    arg_key: str,
    positional_args: dict,
    config: Config,
) -> dict | list | Config | None:
    """Get the value of a given argument key from either the positional arguments or the config.

    Args:
        arg_key (str): The key of the argument to retrieve.
        positional_args (dict): The positional arguments passed to the function.
        config (Config): The configuration object to retrieve the argument from.

    Returns:
        dict | list | None: The value of the argument, or the default value if not found.

    Examples:
    >>> get_value_from_config(
        arg_key="max_epochs",
        positional_args={"max_epochs": 10},
        config=Config({"max_epochs": 20}),
    )
    10
    >>> get_value_from_config(
        arg_key="max_epochs",
        positional_args={},
        config=Config({"max_epochs": 20}),
    )
    20
    """
    # Priority 1: Positional Args
    result = positional_args.get(arg_key, None)
    # Priority 2: Input Config Value
    return config.get(arg_key, None) if result is None else result


def update_train_config(
    train_dataloader: torch.utils.data.DataLoader,
    arguments: dict,
    default_config: Config,
    config: Config,
) -> None:
    """Update the training configuration with the given arguments and default configuration.

    Args:
        train_dataloader (torch.utils.data.DataLoader): The training dataloader.
        arguments (dict): The arguments to update the configuration.
        default_config (Config): The default configuration.
        config (Config): The configuration to update.

    Raises:
        ValueError: If both `max_epochs` and `max_iters` are set.

    Returns:
        None
    """
    config["train_dataloader"] = train_dataloader
    precision = get_value_from_config("precision", arguments, config=default_config)
    max_iters = get_value_from_config("max_iters", arguments, config=default_config)
    max_epochs = get_value_from_config("max_epochs", arguments, config=default_config)
    val_interval = get_value_from_config("val_interval", arguments, config=default_config)
    if max_iters is not None and max_epochs is not None:
        msg = "Only one of `max_epochs` or `max_iters` can be set."
        raise ValueError(msg)
    if "train_cfg" not in config or config["train_cfg"] is None:
        config["train_cfg"] = {"val_interval": val_interval, "by_epoch": True}
    if max_epochs is not None:
        config["train_cfg"]["by_epoch"] = True
        config["train_cfg"]["max_epochs"] = max_epochs
    elif max_iters is not None:
        config["train_cfg"]["by_epoch"] = False
        config["train_cfg"]["max_iters"] = max_iters
    # Update Optimizer
    if "optim_wrapper" not in config or config["optim_wrapper"] is None:
        optimizer = get_value_from_config("optimizer", arguments, config=default_config)
        if get_device() not in ("cuda", "gpu", "npu", "mlu"):
            config["optim_wrapper"] = {"type": "OptimWrapper", "optimizer": optimizer}
        else:
            config["optim_wrapper"] = {
                "type": "AmpOptimWrapper", "dtype": precision, "optimizer": optimizer,
            }
