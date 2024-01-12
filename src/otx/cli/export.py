# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for model export."""
# ruff: noqa

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra import compose, initialize
from jsonargparse import ArgumentParser

from otx.cli.utils.hydra import configure_hydra_outputs

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands


def add_export_parser(subcommands_action: _ActionSubCommands) -> None:
    """Add subparser for export command.

    Args:
        subcommands_action (_ActionSubCommands): Sub-Command in CLI.

    Returns:
        None
    """
    parser = ArgumentParser()
    parser.add_argument("overrides", help="overrides values", default=[], nargs="+")
    subcommands_action.add_subcommand("export", parser, help="Exporting subcommand for OTX")


def otx_export(overrides: list[str]) -> None:
    """Main entry point for model exporting.

    :param overrides: Override List values.
    :return: Optional[str] path to the exported model file.
    """
    from otx.core.config import register_configs

    # This should be in front of hydra.initialize()
    register_configs()

    with initialize(config_path="../config", version_base="1.3", job_name="otx_test"):
        cfg = compose(config_name="test", overrides=overrides, return_hydra_config=True)
        configure_hydra_outputs(cfg)

        # test the model
        from otx.core.engine.export import export

        output_path = export(cfg)
