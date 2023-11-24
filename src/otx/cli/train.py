# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLI entrypoint for training."""
# ruff: noqa

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra import compose, initialize
from jsonargparse import ArgumentParser

from otx.cli.utils.hydra import configure_hydra_outputs
from otx.core.config import register_configs

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands

register_configs()

def add_train_parser(subcommands_action: _ActionSubCommands) -> None:
    """Add subparser for train command.

    Args:
        subcommands_action (_ActionSubCommands): Sub-Command in CLI.

    Returns:
        None
    """
    parser = ArgumentParser()
    parser.add_argument("overrides", help="overrides values", default=[], nargs="+")
    subcommands_action.add_subcommand("train", parser, help="Training subcommand for OTX")


def otx_train(overrides: list[str]) -> None:
    """Main entry point for training.

    :param overrides: Override List values.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # utils.extras(cfg)
    initialize(config_path="../config", version_base="1.3", job_name="otx_train")
    cfg = compose(config_name="train", overrides=overrides, return_hydra_config=True)
    configure_hydra_outputs(cfg)

    # train the model
    from otx.core.engine.train import train
    metric_dict, _ = train(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = utils.get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value
