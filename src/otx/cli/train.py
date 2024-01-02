# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for training."""
# ruff: noqa

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra import compose, initialize
from jsonargparse import ArgumentParser

from otx.cli.utils.hydra import configure_hydra_outputs


if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands
    from pytorch_lightning import Trainer


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


def otx_train(overrides: list[str]) -> dict[str, Any]:
    """Main entry point for training.

    :param overrides: Override List values.
    :return: Metrics values obtained from the model trainer.
    """
    from otx.core.config import register_configs

    # This should be in front of hydra.initialize()
    register_configs()

    with initialize(config_path="../config", version_base="1.3", job_name="otx_train"):
        cfg = compose(config_name="train", overrides=overrides, return_hydra_config=True)
        configure_hydra_outputs(cfg)

        # train the model
        from otx.core.engine import Engine

        engine = Engine()
        return engine.train(cfg)
