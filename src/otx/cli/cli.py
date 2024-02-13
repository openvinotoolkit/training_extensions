"""CLI entrypoints."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from warnings import warn

import yaml
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace, namespace_to_dict
from rich.console import Console

from otx import OTX_LOGO, __version__
from otx.cli.utils.help_formatter import CustomHelpFormatter
from otx.cli.utils.jsonargparse import get_short_docstring, patch_update_configs
from otx.core.types.task import OTXTaskType
from otx.core.utils.imports import get_otx_root_path

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands

_ENGINE_AVAILABLE = True
try:
    from otx.core.config import register_configs
    from otx.engine import Engine

    register_configs()
except ImportError:
    _ENGINE_AVAILABLE = False


class OTXCLI:
    """OTX CLI entrypoint."""

    def __init__(self) -> None:
        """Initialize OTX CLI."""
        self.console = Console()
        self._subcommand_method_arguments: dict[str, list[str]] = {}
        with patch_update_configs():
            self.parser = self.init_parser()
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
            parser_mode="omegaconf",
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

    @staticmethod
    def engine_subcommand_parser(**kwargs) -> ArgumentParser:
        """Creates an ArgumentParser object for the engine subcommand.

        Args:
            **kwargs: Additional keyword arguments to be passed to the ArgumentParser constructor.

        Returns:
            ArgumentParser: The created ArgumentParser object.
        """
        parser = ArgumentParser(
            formatter_class=CustomHelpFormatter,
            parser_mode="omegaconf",
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
            type=Optional[str],
            help="Path to dataset root.",
        )
        parser.add_argument(
            "--task",
            type=str,
            help="Task Type.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Sets seed for pseudo-random number generators in: pytorch, numpy, python.random.",
        )
        parser.add_argument(
            "--callback_monitor",
            type=str,
            help="The metric to monitor the model performance during training callbacks.",
        )
        parser.add_argument(
            "--disable-infer-num-classes",
            help="OTX automatically infers num_classes from the given dataset "
            "and applies it to the model initialization."
            "Consequently, there might be a mismatch with the provided model configuration during runtime. "
            "Setting this option to true will disable this behavior.",
            action="store_true",
        )
        engine_skip = {"model", "datamodule", "optimizer", "scheduler"}
        parser.add_class_arguments(
            Engine,
            "engine",
            fail_untyped=False,
            sub_configs=True,
            instantiate=False,
            skip=engine_skip,
        )
        # Model Settings
        from otx.core.model.entity.base import OTXModel

        model_kwargs: dict[str, Any] = {"fail_untyped": False}

        parser.add_subclass_arguments(
            OTXModel,
            "model",
            required=False,
            **model_kwargs,
        )
        # Datamodule Settings
        from otx.core.data.module import OTXDataModule

        parser.add_class_arguments(
            OTXDataModule,
            "data",
            fail_untyped=False,
            sub_configs=True,
        )
        # Optimizer & Scheduler Settings
        from lightning.pytorch.cli import ReduceLROnPlateau
        from torch.optim import Optimizer
        from torch.optim.lr_scheduler import LRScheduler

        optim_kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"params"}}
        scheduler_kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"optimizer"}}
        parser.add_subclass_arguments(
            baseclass=(Optimizer, list[Optimizer]),
            nested_key="optimizer",
            **optim_kwargs,
        )
        scheduler_type = (LRScheduler, ReduceLROnPlateau, list[LRScheduler | ReduceLROnPlateau])
        parser.add_subclass_arguments(
            baseclass=scheduler_type,
            nested_key="scheduler",
            **scheduler_kwargs,
        )

        return parser

    @staticmethod
    def engine_subcommands() -> dict[str, set[str]]:
        """Returns dictionary the subcommands of engine, and whose value is the argument to be skipped in the CLI.

        This allows the CLI to skip duplicate keys when creating the Engine and when running the subcommand.

        Returns:
            A dictionary where the keys are the subcommands and the values are sets of skipped arguments.
        """
        device_kwargs = {"accelerator", "devices"}
        return {
            "train": {"seed"}.union(device_kwargs),
            "test": {"datamodule"}.union(device_kwargs),
            "predict": {"datamodule"}.union(device_kwargs),
            "export": device_kwargs,
            "optimize": {"datamodule"}.union(device_kwargs),
            "explain": {"datamodule"}.union(device_kwargs),
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
            # If environment is not configured to use Engine, do not add a subcommand for Engine.
            return
        for subcommand in self.engine_subcommands():
            parser_kwargs = self._set_default_config_from_auto_configurator()
            sub_parser = self.engine_subcommand_parser(**parser_kwargs)

            sub_parser.link_arguments("data_root", "engine.data_root")
            sub_parser.link_arguments("data_root", "data.config.data_root")

            fn = getattr(Engine, subcommand)
            description = get_short_docstring(fn)

            added_arguments = sub_parser.add_method_arguments(
                Engine,
                subcommand,
                skip=set(self.engine_subcommands()[subcommand]),
                fail_untyped=False,
            )

            if "logger" in added_arguments:
                sub_parser.link_arguments("engine.work_dir", "logger.init_args.save_dir")
            if "callbacks" in added_arguments:
                sub_parser.link_arguments("callback_monitor", "callbacks.init_args.monitor")
                sub_parser.link_arguments("engine.work_dir", "callbacks.init_args.dirpath")
            if "checkpoint" in added_arguments and "--checkpoint" in sys.argv:
                # This is code for an OVModel that uses checkpoint in model.model_name.
                sub_parser.link_arguments("checkpoint", "model.init_args.model_name")

            # Load default subcommand config file
            default_config_file = get_otx_root_path() / "recipe" / "_base_" / f"{subcommand}.yaml"
            if default_config_file.exists():
                with Path(default_config_file).open() as f:
                    default_config = yaml.safe_load(f)
                sub_parser.set_defaults(**default_config)

            self._subcommand_method_arguments[subcommand] = added_arguments
            self._subcommand_parsers[subcommand] = sub_parser
            parser_subcommands.add_subcommand(subcommand, sub_parser, help=description)

    def _set_default_config_from_auto_configurator(self) -> dict:
        parser_kwargs = {}
        data_root = None
        task = None
        if "--data_root" in sys.argv:
            data_root = sys.argv[sys.argv.index("--data_root") + 1]
        if "--task" in sys.argv:
            task = sys.argv[sys.argv.index("--task") + 1]
        enable_auto_config = data_root is not None and "--config" not in sys.argv
        if enable_auto_config:
            from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK, AutoConfigurator

            auto_configurator = AutoConfigurator(data_root=data_root, task=OTXTaskType(task))
            config_file_path = DEFAULT_CONFIG_PER_TASK[auto_configurator.task]
            parser_kwargs["default_config_files"] = [config_file_path]
        return parser_kwargs

    def _set_extension_subcommands_parser(self, parser_subcommands: _ActionSubCommands) -> None:
        from otx.cli.install import add_install_parser

        add_install_parser(parser_subcommands)

        if _ENGINE_AVAILABLE:
            # `otx find` arguments
            find_parser = ArgumentParser(formatter_class=CustomHelpFormatter)
            find_parser.add_argument(
                "--task",
                help="Value for filtering by task. Default is None, which shows all recipes.",
                type=Optional[OTXTaskType],
            )
            find_parser.add_argument(
                "--pattern",
                help="This allows you to filter the model name of the recipe. \
                      For example, if you want to find all models that contain the word 'efficient', \
                      you can use '--pattern efficient'",
                type=Optional[str],
            )
            parser_subcommands.add_subcommand("find", find_parser, help="This shows the model provided by OTX.")

    def instantiate_classes(self) -> None:
        """Instantiate the necessary classes based on the subcommand.

        This method checks if the subcommand is one of the engine subcommands.
        If it is, it instantiates the necessary classes such as config, datamodule, model, and engine.
        """
        if self.subcommand in self.engine_subcommands():
            # For num_classes update, Model is instantiated separately.
            model_config = self.config[self.subcommand].pop("model")
            self.config_init = self.parser.instantiate_classes(self.config)
            self.datamodule = self.get_config_value(self.config_init, "data")
            self.model, optimizer, scheduler = self.instantiate_model(model_config=model_config)

            engine_kwargs = self.get_config_value(self.config_init, "engine")
            self.engine = Engine(
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                datamodule=self.datamodule,
                **engine_kwargs,
            )

    def instantiate_model(self, model_config: Namespace) -> tuple:
        """Instantiate the model based on the subcommand.

        This method checks if the subcommand is one of the engine subcommands.
        If it is, it instantiates the model.

        Args:
            model_config (Namespace): The model configuration.

        Returns:
            tuple: The model and optimizer and scheduler.
        """
        from otx.core.model.entity.base import OTXModel

        # Update num_classes
        if not self.get_config_value(self.config_init, "disable_infer_num_classes", False):
            num_classes = self.datamodule.meta_info.num_classes
            if num_classes != model_config.init_args.num_classes:
                warning_msg = (
                    f"The `num_classes` in dataset is {num_classes} "
                    f"but, the `num_classes` of model is {model_config.init_args.num_classes}. "
                    f"So, Update `model.num_classes` to {num_classes}."
                )
                warn(warning_msg, stacklevel=0)
                model_config.init_args.num_classes = num_classes

        # Parses the OTXModel separately to update num_classes.
        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(OTXModel, "model", required=False, fail_untyped=False)
        model = model_parser.instantiate_classes(Namespace(model=model_config)).get("model")

        # Update self.config with model
        self.config[self.subcommand].update(Namespace(model=model_config))

        from otx.core.utils.instantiators import partial_instantiate_class

        optimizer_kwargs = self.get_config_value(self.config_init, "optimizer", Namespace())
        optimizer_kwargs = optimizer_kwargs if isinstance(optimizer_kwargs, list) else [optimizer_kwargs]
        optimizers = partial_instantiate_class([namespace_to_dict(_opt) for _opt in optimizer_kwargs])

        scheduler_kwargs = self.get_config_value(self.config_init, "scheduler", Namespace())
        scheduler_kwargs = scheduler_kwargs if isinstance(scheduler_kwargs, list) else [scheduler_kwargs]
        schedulers = partial_instantiate_class([namespace_to_dict(_sch) for _sch in scheduler_kwargs])

        return model, optimizers, schedulers

    def get_config_value(self, config: Namespace, key: str, default: Any = None) -> Any:  # noqa: ANN401
        """Retrieves the value of a configuration key from the given config object.

        Args:
            config (Namespace): The config object containing the configuration values.
            key (str): The key of the configuration value to retrieve.
            default (Any, optional): The default value to return if the key is not found. Defaults to None.

        Returns:
            Any: The value of the configuration key, or the default value if the key is not found.
        """
        return config.get(str(self.subcommand), config).get(key, default)

    def get_subcommand_parser(self, subcommand: str | None) -> ArgumentParser:
        """Returns the argument parser for the specified subcommand.

        Args:
            subcommand (str | None): The name of the subcommand. If None, returns the main parser.

        Returns:
            ArgumentParser: The argument parser for the specified subcommand.
        """
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
        """
        self.get_subcommand_parser(self.subcommand).save(
            cfg=self.config.get(str(self.subcommand), self.config),
            path=Path(self.engine.work_dir) / "configs.yaml",
            overwrite=True,
            multifile=False,
            skip_check=True,
        )

    def set_seed(self) -> None:
        """Set the random seed for reproducibility.

        This method retrieves the seed value from the argparser and uses it to set the random seed.
        If a seed value is provided, it will be used to set the random seed using the
        `seed_everything` function from the `lightning` module.
        """
        seed = self.get_config_value(self.config, "seed", None)
        if seed is not None:
            from lightning import seed_everything

            seed_everything(seed, workers=True)

    def run(self) -> None:
        """Executes the specified subcommand.

        Raises:
            ValueError: If the subcommand is not recognized.
        """
        self.console.print(f"[blue]{OTX_LOGO}[/blue] ver.{__version__}", justify="center")
        if self.subcommand == "install":
            from otx.cli.install import otx_install

            otx_install(**self.config["install"])
        elif self.subcommand == "find":
            from otx.engine.utils.api import list_models

            list_models(print_table=True, **self.config[self.subcommand])
        elif self.subcommand in self.engine_subcommands():
            self.set_seed()
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
