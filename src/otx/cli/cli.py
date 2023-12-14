# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from otx import __version__
from otx.cli.utils.jsonargparse import get_short_docstring, update

# [FIXME]: Overriding Namespce.update to match mmengine.Config (DictConfig | dict)
# and prevent int, float types from being converted to str
# https://github.com/omni-us/jsonargparse/issues/236
Namespace.update = update

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands

_ENGINE_AVAILABLE = True
try:
    from otx.core.engine.engine import Engine
except ImportError:
    _ENGINE_AVAILABLE = False


class OTXCLI:
    """OTX CLI entrypoint."""

    def __init__(self) -> None:
        """Initialize OTX CLI."""
        self.parser = self.init_parser()
        self._subcommand_method_arguments: dict[str, list[str]] = {}
        self.add_subcommands()
        self.config = self.parser.parse_args()

        self.subcommand = self.config["subcommand"]
        self.run()

    def init_parser(self, **kwargs) -> ArgumentParser:
        """Initialize the argument parser for the OTX CLI.

        Args:
            **kwargs: Additional keyword arguments to pass to the ArgumentParser constructor.

        Returns:
            ArgumentParser: The initialized argument parser.
        """
        parser = ArgumentParser(
            description="OpenVINO Training-Extension command line tool",
            env_prefix="otx",
        )
        parser.add_argument(
            "-V",
            "--version",
            action="version",
            version=f"%(prog)s {__version__}",
            help="Display OTX version number.",
        )
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        return parser

    @staticmethod
    def engine_subcommands() -> dict[str, set[str]]:
        """Returns a dictionary of engine subcommands and their required arguments.

        Returns:
            A dictionary where the keys are the subcommands and the values are sets of required arguments.
        """
        return {
            "train": {"model", "datamodule"},
            "test": {"model", "datamodule"},
            "predict": {"model"},
            "export": {"model"},
        }

    def add_subcommands(self) -> None:
        """Adds subcommands to the CLI parser.

        This method initializes and configures subcommands for the OTX CLI parser.
        It iterates over the available subcommands, adds arguments specific to each subcommand,
        and registers them with the parser.

        Returns:
            None
        """
        self._subcommand_parsers: dict[str, ArgumentParser] = {}
        parser_subcommands = self.parser.add_subcommands()
        self._set_extension_subcommands_parser(parser_subcommands)
        if not _ENGINE_AVAILABLE:
            return
        for subcommand in self.engine_subcommands():
            sub_parser = ArgumentParser()
            sub_parser.add_argument(
                "-c",
                "--config",
                action=ActionConfigFile,
                help="Path to a configuration file in json or yaml format.",
            )
            sub_parser.add_class_arguments(
                Engine,
                "engine",
                fail_untyped=False,
                sub_configs=True,
            )
            if "model" in self.engine_subcommands()[subcommand]:
                from otx.core.model.module.base import OTXLitModule
                sub_parser.add_subclass_arguments(
                    OTXLitModule,
                    "model",
                    fail_untyped=False,
                    required=True,
                )
            if "datamodule" in self.engine_subcommands()[subcommand]:
                from otx.core.data.module import OTXDataModule
                sub_parser.add_class_arguments(
                    OTXDataModule,
                    "data",
                    fail_untyped=False,
                    sub_configs=True,
                )
            skip: set[str | int] = set(self.engine_subcommands()[subcommand])
            fn = getattr(Engine, subcommand)
            description = get_short_docstring(fn)
            added = sub_parser.add_method_arguments(
                Engine, subcommand, skip=skip, fail_untyped=False,
            )
            self._subcommand_method_arguments[subcommand] = added
            self._subcommand_parsers[subcommand] = sub_parser
            parser_subcommands.add_subcommand(subcommand, sub_parser, help=description)

    def _set_extension_subcommands_parser(self, parser_subcommands: _ActionSubCommands) -> None:
        from otx.cli.install import add_install_parser
        add_install_parser(parser_subcommands)

    def instantiate_classes(self) -> None:
            """Instantiate the necessary classes based on the subcommand.

            This method checks if the subcommand is one of the engine subcommands.
            If it is, it instantiates the necessary classes such as config, datamodule, model, and engine.
            """
            if self.subcommand in self.engine_subcommands():
                from otx.core.engine.engine import Engine
                self.config_init = self.parser.instantiate_classes(self.config)
                self.datamodule = self._get(self.config_init, "data")
                self.model = self._get(self.config_init, "model")
                self.engine = Engine()

    def _get(self, config: Namespace, key: str, default: Any = None) -> Any:  # noqa: ANN401
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(self.subcommand), config).get(key, default)

    def _prepare_subcommand_kwargs(self, subcommand: str) -> dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        fn_kwargs = {
            k: v for k, v in self.config_init[subcommand].items() if k in self._subcommand_method_arguments[subcommand]
        }
        fn_kwargs["model"] = self.model
        if self.datamodule is not None:
            fn_kwargs["datamodule"] = self.datamodule
        return fn_kwargs

    def run(self) -> None:
        """Executes the specified subcommand.

        Raises:
            ValueError: If the subcommand is not recognized.
        """
        if self.subcommand == "install":
            from otx.cli.install import otx_install
            otx_install(**self.config["install"])
        elif self.subcommand in self.engine_subcommands():
            self.instantiate_classes()
            fn_kwargs = self._prepare_subcommand_kwargs(self.subcommand)
            fn = getattr(self.engine, self.subcommand)
            fn(**fn_kwargs)
        else:
            msg = f"Unrecognized subcommand: {self.subcommand}"
            raise ValueError(msg)


def main() -> None:
    """Entry point for OTX CLI.

    This function is a single entry point for all OTX CLI related operations:
    """
    OTXCLI()


if __name__ == "__main__":
    main()
