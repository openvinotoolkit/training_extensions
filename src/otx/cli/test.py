# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for testing."""
# ruff: noqa

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra import compose, initialize
from jsonargparse import ArgumentParser

from otx.cli.utils.hydra import configure_hydra_outputs

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands


def add_test_parser(subcommands_action: _ActionSubCommands) -> None:
    """Add subparser for test command.

    Args:
        subcommands_action (_ActionSubCommands): Sub-Command in CLI.

    Returns:
        None
    """
    parser = ArgumentParser()
    parser.add_argument("overrides", help="overrides values", default=[], nargs="+")
    subcommands_action.add_subcommand("test", parser, help="Testing subcommand for OTX")


def otx_test(overrides: list[str]) -> None:
    """Main entry point for testing.

    :param overrides: Override List values.
    :return: Optional[float] with optimized metric value.
    """
    from otx.core.config import register_configs

    # This should be in front of hydra.initialize()
    register_configs()

    with initialize(config_path="../config", version_base="1.3", job_name="otx_test"):
        cfg = compose(config_name="test", overrides=overrides, return_hydra_config=True)
        configure_hydra_outputs(cfg)

        # test the model
        from otx.core.engine.test import test

        metric_dict, _ = test(cfg)
