"""CLI entrypoints."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace, namespace_to_dict
from rich.console import Console

from otx import OTX_LOGO, __version__
from otx.cli.utils import get_otx_root_path
from otx.cli.utils.help_formatter import CustomHelpFormatter
from otx.cli.utils.jsonargparse import get_short_docstring

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands

_ENGINE_AVAILABLE = True
try:
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
        parser.add_argument(
            "--callback_monitor",
            type=str,
            help="The metric for that model to monitor in the callback.",
        )
        return parser

    @staticmethod
    def engine_subcommands() -> dict[str, set[str]]:
        """Returns a dictionary of engine subcommands and their required arguments.

        Returns:
            A dictionary where the keys are the subcommands and the values are sets of required arguments.
        """
        device_kwargs = {"accelerator", "devices"}
        return {
            "train": device_kwargs,
            "test": {"datamodule"}.union(device_kwargs),
            "predict": {"datamodule"}.union(device_kwargs),
            "export": device_kwargs,
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
            sub_parser = self.subcommand_parser()
            engine_skip = {"model", "datamodule", "optimizer", "scheduler"}
            sub_parser.add_class_arguments(
                Engine,
                "engine",
                fail_untyped=False,
                sub_configs=True,
                instantiate=False,
                skip=engine_skip,
            )
            sub_parser.link_arguments("data_root", "engine.data_root")

            # Model Settings
            from otx.core.model.entity.base import OTXModel

            model_kwargs: dict[str, Any] = {"fail_untyped": False}

            sub_parser.add_subclass_arguments(
                OTXModel,
                "model",
                required=False,
                **model_kwargs,
            )
            # Datamodule Settings
            from otx.core.data.module import OTXDataModule

            sub_parser.add_class_arguments(
                OTXDataModule,
                "data",
                fail_untyped=False,
                sub_configs=True,
            )
            sub_parser.link_arguments("data_root", "data.config.data_root")

            # Optimizer & Scheduler Settings
            from lightning.pytorch.cli import LRSchedulerTypeTuple
            from torch.optim import Optimizer

            optim_kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"params"}}
            scheduler_kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"optimizer"}}
            sub_parser.add_subclass_arguments(
                baseclass=(Optimizer,),
                nested_key="optimizer",
                **optim_kwargs,
            )
            sub_parser.add_subclass_arguments(
                baseclass=LRSchedulerTypeTuple,
                nested_key="scheduler",
                **scheduler_kwargs,
            )

            skip: set[str | int] = set(self.engine_subcommands()[subcommand])
            fn = getattr(Engine, subcommand)
            description = get_short_docstring(fn)
            added = sub_parser.add_method_arguments(
                Engine,
                subcommand,
                skip=skip,
                fail_untyped=False,
            )

            # Load default subcommand config file
            default_config_file = Path(get_otx_root_path()) / "recipe" / "_base_" / f"{subcommand}.yaml"
            if default_config_file.exists():
                with Path(default_config_file).open() as f:
                    default_config = yaml.safe_load(f)
                sub_parser.set_defaults(**default_config)

            if "logger" in added:
                sub_parser.link_arguments("engine.work_dir", "logger.init_args.save_dir")
            if "callbacks" in added:
                sub_parser.link_arguments("callback_monitor", "callbacks.init_args.monitor")
                sub_parser.link_arguments("engine.work_dir", "callbacks.init_args.dirpath")

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
            self.model, optimizer, scheduler = self.instantiate_model()

            engine_kwargs = self._get(self.config_init, "engine")
            self.engine = Engine(
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                datamodule=self.datamodule,
                **engine_kwargs,
            )

    def instantiate_model(self) -> tuple:
        """Instantiate the model based on the subcommand.

        This method checks if the subcommand is one of the engine subcommands.
        If it is, it instantiates the model.

        Returns:
            tuple: The model and optimizer and scheduler.
        """
        model = self._get(self.config_init, "model")
        optimizer_kwargs = self._get(self.config_init, "optimizer", {})
        if isinstance(optimizer_kwargs, Namespace):
            optimizer_kwargs = namespace_to_dict(optimizer_kwargs)
        scheduler_kwargs = self._get(self.config_init, "scheduler", {})
        if isinstance(scheduler_kwargs, Namespace):
            scheduler_kwargs = namespace_to_dict(scheduler_kwargs)
        from otx.core.utils.instantiators import partial_instantiate_class

        return model, partial_instantiate_class(optimizer_kwargs), partial_instantiate_class(scheduler_kwargs)

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
        return {
            k: v for k, v in self.config_init[subcommand].items() if k in self._subcommand_method_arguments[subcommand]
        }

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
