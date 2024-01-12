# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoints."""

from __future__ import annotations

from jsonargparse import ArgumentParser

from otx import __version__


class OTXCLI:
    """OTX CLI entrypoint."""

    def __init__(self) -> None:
        """Initialize OTX CLI."""
        self.parser = self.init_parser()
        self.setup_subcommands()
        self.config = self.parser.parse_args()

        self.run()

    def init_parser(self, **kwargs) -> ArgumentParser:
        """Initialize the argument parser for the OTX CLI.

        Args:
            **kwargs: Additional keyword arguments to pass to the ArgumentParser constructor.

        Returns:
            ArgumentParser: The initialized argument parser.
        """
        parser = ArgumentParser(description="OpenVINO Training-Extension command line tool", env_prefix="otx")
        parser.add_argument(
            "-V",
            "--version",
            action="version",
            version=f"%(prog)s {__version__}",
            help="Display OTX version number.",
        )
        return parser

    def setup_subcommands(self) -> None:
        """Setup subcommands for the OTX CLI."""
        parser_subcommands = self.parser.add_subcommands()
        from otx.cli.install import add_install_parser

        add_install_parser(parser_subcommands)
        from otx.cli.train import add_train_parser

        add_train_parser(parser_subcommands)
        from otx.cli.test import add_test_parser

        add_test_parser(parser_subcommands)
        from otx.cli.export import add_export_parser

        add_export_parser(parser_subcommands)

    def run(self) -> None:
        """Run the OTX CLI."""
        subcommand = self.config["subcommand"]
        if subcommand == "install":
            from otx.cli.install import otx_install

            otx_install(**self.config["install"])
        elif subcommand == "train":
            from otx.cli.train import otx_train

            otx_train(**self.config["train"])
        elif subcommand == "test":
            from otx.cli.test import otx_test

            otx_test(**self.config["test"])

        elif subcommand == "export":
            from otx.cli.export import otx_export

            otx_export(**self.config["export"])


def main() -> None:
    """Entry point for OTX CLI.

    This function is a single entry point for all OTX CLI related operations:
    """
    OTXCLI()


if __name__ == "__main__":
    main()
