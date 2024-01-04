"""CLI entrypoints."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from rich.console import Console

from otx import OTX_LOGO, __version__
from otx.cli.utils.help_formatter import CustomHelpFormatter
from otx.cli.utils.jsonargparse import flatten_dict, get_short_docstring

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands

_ENGINE_AVAILABLE = True
try:
    from otx.core.utils.auto_configuration import AutoConfigurator
    from otx.engine import Engine
except ImportError:
    _ENGINE_AVAILABLE = False


class OTXCLI:
    """OTX CLI entrypoint."""

    def __init__(self) -> None:
        """Initialize OTX CLI."""
        self.console = Console()
        self.parser = self.init_parser()
        self._subcommand_method_arguments: dict[str, list[str]] = {}
        self.add_subcommands()
        self.config = self.parser.parse_args(_skip_check=True)

        self.subcommand = self.config["subcommand"]
        self.run()

    def init_parser(self) -> ArgumentParser:
        """Initialize the argument parser for the OTX CLI.

        Returns:
            ArgumentParser: The initialized argument parser.
        """
        parser = ArgumentParser(
            description="OpenVINO Training-Extension command line tool",
            env_prefix="otx",
            formatter_class=CustomHelpFormatter,
        )
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"%(prog)s {__version__}",
            help="Display OTX version number.",
        )
        return parser

    def subcommand_parser(self, **kwargs) -> ArgumentParser:
        """Returns an ArgumentParser object for parsing command line arguments specific to a subcommand.

        Returns:
            ArgumentParser: An ArgumentParser object configured with the specified arguments.
        """
        parser = ArgumentParser(
            formatter_class=CustomHelpFormatter,
            **kwargs,
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            help="Verbose mode. This shows a configuration argument that allows for more specific overrides. \
                Multiple -v options increase the verbosity. The maximum is 2.",
        )
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        parser.add_argument(
            "--data_root",
            type=str,
            help="Path to dataset root.",
        )
        parser.add_argument(
            "--task",
            type=str,
            help="Task Type.",
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
            # Auto-Configuration
            data_root = None
            task = None
            if "--data_root" in sys.argv:
                data_root = sys.argv[sys.argv.index("--data_root") + 1]
            if "--task" in sys.argv:
                task = sys.argv[sys.argv.index("--task") + 1]
            auto_configurator = AutoConfigurator(data_root=data_root, task=task)

            sub_parser = self.subcommand_parser()
            sub_parser.add_class_arguments(
                Engine,
                "engine",
                fail_untyped=False,
                sub_configs=True,
            )
            sub_parser.link_arguments("data_root", "engine.data_root")

            if "model" in self.engine_subcommands()[subcommand]:
                from otx.core.model.module.base import OTXLitModule

                model_kwargs: dict[str, Any] = {"fail_untyped": False}
                if data_root is not None and "--model" not in sys.argv:
                    # Add Default values from Auto-Configurator
                    model_kwargs["default"] = auto_configurator.load_default_model_config()

                sub_parser.add_subclass_arguments(
                    OTXLitModule,
                    "model",
                    required=False,
                    **model_kwargs,
                )
            if "datamodule" in self.engine_subcommands()[subcommand]:
                from otx.core.data.module import OTXDataModule

                sub_parser.add_class_arguments(
                    OTXDataModule,
                    "data",
                    fail_untyped=False,
                    sub_configs=True,
                )

                if data_root is not None:
                    # Add Default values from Auto-Configurator
                    default_data_config = auto_configurator.load_default_data_config()
                    default_data_config = flatten_dict({"data": default_data_config})
                    default_data_config["data.config.data_root"] = data_root
                    default_data_config["data.task"] = task if task is not None else auto_configurator.task
                    sub_parser.set_defaults(default_data_config)

            skip: set[str | int] = set(self.engine_subcommands()[subcommand])
            fn = getattr(Engine, subcommand)
            description = get_short_docstring(fn)
            added = sub_parser.add_method_arguments(
                Engine,
                subcommand,
                skip=skip,
                fail_untyped=False,
            )
            if data_root is not None and subcommand == "train":
                # Add Default values from Auto-Configurator
                default_engine_config = auto_configurator.load_default_engine_config()
                default_engine_config = flatten_dict(default_engine_config)
                default_engine_config["engine.data_root"] = data_root
                default_engine_config["engine.task"] = task if task is not None else auto_configurator.task
                sub_parser.set_defaults(default_engine_config)

            if "logger" in added:
                sub_parser.link_arguments("engine.work_dir", "logger.init_args.save_dir")

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
            self.config_init = self.parser.instantiate_classes(self.config)
            self.datamodule = self._get(self.config_init, "data")
            self.model = self._get(self.config_init, "model")
            self.engine = self._get(self.config_init, "engine")

    def _get(self, config: Namespace, key: str, default: Any = None) -> Any:  # noqa: ANN401
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(self.subcommand), config).get(key, default)

    def _parser(self, subcommand: str | None) -> ArgumentParser:
        if subcommand is None:
            return self.parser
        # return the subcommand parser for the subcommand passed
        return self._subcommand_parsers[subcommand]

    def _prepare_subcommand_kwargs(self, subcommand: str) -> dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        fn_kwargs = {
            k: v for k, v in self.config_init[subcommand].items() if k in self._subcommand_method_arguments[subcommand]
        }
        fn_kwargs["model"] = self.model
        if self.datamodule is not None:
            fn_kwargs["datamodule"] = self.datamodule
        return fn_kwargs

    def save_config(self) -> None:
        """Save the configuration for the specified subcommand.

        The configuration is saved as a YAML file in the engine's working directory.

        Args:
            None

        Returns:
            None
        """
        self._parser(self.subcommand).save(
            cfg=self.config.get(str(self.subcommand), self.config),
            path=Path(self.engine.work_dir) / "configs.yaml",
            overwrite=True,
            multifile=False,
            skip_check=True,
        )

    def run(self) -> None:
        """Executes the specified subcommand.

        Raises:
            ValueError: If the subcommand is not recognized.
        """
        self.console.print(f"[blue]{OTX_LOGO}[/blue] ver.{__version__}", justify="center")
        if self.subcommand == "install":
            from otx.cli.install import otx_install

            otx_install(**self.config["install"])
        elif self.subcommand in self.engine_subcommands():
            self.instantiate_classes()
            fn_kwargs = self._prepare_subcommand_kwargs(self.subcommand)
            fn = getattr(self.engine, self.subcommand)
            try:
                fn(**fn_kwargs)
            except Exception:
                self.console.print_exception(width=self.console.width)
            self.save_config()
        else:
            msg = f"Unrecognized subcommand: {self.subcommand}"
            raise ValueError(msg)
