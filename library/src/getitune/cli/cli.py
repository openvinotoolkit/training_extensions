# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoints."""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from warnings import warn

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace, namespace_to_dict
from rich.console import Console

from getitune import LIBRARY_LOGO, __version__
from getitune.cli.utils import absolute_path
from getitune.cli.utils.help_formatter import CustomHelpFormatter
from getitune.cli.utils.jsonargparse import get_short_docstring, patch_update_configs
from getitune.cli.utils.workspace import Workspace
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands

    from getitune.backend.lightning.models.base import LightningModel
    from getitune.data.module import DataModule


_ENGINE_AVAILABLE = True
try:
    from getitune.backend.lightning.engine import LightningEngine
    from getitune.config import register_configs

    register_configs()
except ImportError:
    _ENGINE_AVAILABLE = False


class CLI:
    """getitune CLI entrypoint."""

    datamodule: DataModule

    def __init__(self, args: list[str] | None = None, run: bool = True) -> None:
        """Initialize getitune CLI."""
        self.console = Console()
        self._subcommand_method_arguments: dict[str, list[str]] = {}
        # Preserve the explicit args passed in so that add_subcommands() and
        # _set_default_config() do not fall back to the global sys.argv when
        # the CLI is constructed programmatically (e.g. by the benchmark
        # runner via LightningEngine.from_config / get_instantiated_classes).
        # Using sys.argv in that case caused stale `.latest/train` artefacts
        # from a previous experiment to be picked up as "defaults" for the
        # current experiment's test/export/optimize phases.
        self._argv: list[str] = list(args) if args is not None else list(sys.argv)
        with patch_update_configs():
            self.parser = self.init_parser()
            self.add_subcommands()
            self.config = self.parser.parse_args(args=args, _skip_check=True)

        self.subcommand = self.config["subcommand"]
        if run:
            self.run()

    @property
    def _effective_argv(self) -> list[str]:
        """Return the effective argv, falling back to sys.argv if not set."""
        return getattr(self, "_argv", None) or list(sys.argv)

    def _argv_contains(self, flag: str) -> bool:
        """Return True if *flag* is present in the effective argv."""
        return flag in self._effective_argv

    def _argv_value_after(self, flag: str) -> str | None:
        """Return the token immediately following *flag* in the effective argv."""
        argv = self._effective_argv
        if flag not in argv:
            return None
        idx = argv.index(flag)
        if idx + 1 >= len(argv):
            return None
        return argv[idx + 1]

    def init_parser(self) -> ArgumentParser:
        """Initialize the argument parser for the getitune CLI.

        Returns:
            ArgumentParser: The initialized argument parser.
        """
        parser = ArgumentParser(
            description="OpenVINO Training-Extension command line tool",
            env_prefix="getitune",
            parser_mode="omegaconf",
            formatter_class=CustomHelpFormatter,
        )
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"%(prog)s {__version__}",
            help="Display getitune version number.",
        )
        return parser

    @staticmethod
    def engine_subcommand_parser(subcommand: str, **kwargs) -> tuple[ArgumentParser, list]:
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
            type=absolute_path,
            help="Path to dataset root.",
        )
        parser.add_argument(
            "--work_dir",
            type=absolute_path,
            default=absolute_path(Path.cwd()),
            help="Path to work directory. The default is created as getitune-workspace.",
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
            help="getitune automatically infers num_classes from the given dataset "
            "and applies it to the model initialization."
            "Consequently, there might be a mismatch with the provided model configuration during runtime. "
            "Setting this option to true will disable this behavior.",
            action="store_true",
        )
        engine_skip = {"model", "work_dir", "data"}
        parser.add_class_arguments(
            LightningEngine,
            "engine",
            fail_untyped=False,
            sub_configs=True,
            instantiate=False,
            skip=engine_skip,
        )
        # Model Settings
        from getitune.backend.lightning.models.base import LightningModel
        from getitune.backend.openvino.models import OVModel

        parser.add_subclass_arguments(
            (LightningModel, OVModel),
            "model",
            required=False,
            fail_untyped=False,
        )
        # Datamodule Settings
        from getitune.data.module import DataModule

        parser.add_class_arguments(
            DataModule,
            "data",
            fail_untyped=False,
            sub_configs=True,
        )

        parser.add_class_arguments(Workspace, "workspace")
        parser.link_arguments("work_dir", "workspace.work_dir")

        parser.link_arguments("data_root", "data.data_root")
        parser.link_arguments("engine.device", "data.device")

        added_arguments = parser.add_method_arguments(
            LightningEngine,
            subcommand,
            skip=set(CLI.engine_subcommands()[subcommand]),
            fail_untyped=False,
        )

        if "callbacks" in added_arguments:
            parser.link_arguments("callback_monitor", "callbacks.init_args.monitor")
            parser.link_arguments("workspace.work_dir", "callbacks.init_args.dirpath", apply_on="instantiate")
        if "logger" in added_arguments:
            parser.link_arguments("workspace.work_dir", "logger.init_args.save_dir", apply_on="instantiate")
            parser.link_arguments("workspace.work_dir", "logger.init_args.log_dir", apply_on="instantiate")
        if (
            "checkpoint" in added_arguments
            and "--checkpoint" in sys.argv
            and any("openvino_model.yaml" in arg for arg in sys.argv)
        ):
            # This is code for an OVModel that uses checkpoint in model.model_name.
            parser.link_arguments("checkpoint", "model.init_args.model_name")

        return parser, added_arguments

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
            "benchmark": device_kwargs,
        }

    def add_subcommands(self) -> None:
        """Adds subcommands to the CLI parser.

        This method initializes and configures subcommands for the getitune CLI parser.
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
            # If already have a workspace or run it from the root of a workspace, utilize config and checkpoint in cache
            work_dir_override = self._argv_value_after("--work_dir")
            root_dir = Path(work_dir_override) if work_dir_override is not None else Path.cwd()
            self.cache_dir = root_dir / ".latest" / "train"  # The config and checkpoint used in the latest training.

            parser_kwargs = self._set_default_config()
            sub_parser, added_arguments = self.engine_subcommand_parser(subcommand=subcommand, **parser_kwargs)
            if not self._argv_contains("--config") and "checkpoint" in added_arguments and self.cache_dir.exists():
                # If the user specifies the config directly, not set the cache ckpt as default.
                self._load_cache_ckpt(parser=sub_parser)

            fn = getattr(LightningEngine, subcommand)
            description = get_short_docstring(fn)

            self._subcommand_method_arguments[subcommand] = added_arguments
            self._subcommand_parsers[subcommand] = sub_parser
            parser_subcommands.add_subcommand(subcommand, sub_parser, help=description)

    def _load_cache_ckpt(self, parser: ArgumentParser) -> None:
        checkpoint_dir = self.cache_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return
        ckpt_files = list(checkpoint_dir.glob("epoch_*.ckpt"))
        if not ckpt_files:
            return
        latest_checkpoint = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        parser.set_defaults(checkpoint=str(latest_checkpoint))
        if not self._argv_contains("--print_config"):
            warn(f"Load default checkpoint from {latest_checkpoint}.", stacklevel=0)

    def _set_default_config(self) -> dict:
        parser_kwargs: dict = {}
        if not self._argv_contains("--config") and (self.cache_dir / "configs.yaml").exists():
            parser_kwargs["default_config_files"] = [str(self.cache_dir / "configs.yaml")]
            if not self._argv_contains("--print_config"):
                warn(f"Load default config from {self.cache_dir / 'configs.yaml'}.", stacklevel=0)
            return parser_kwargs

        return parser_kwargs

    def _set_extension_subcommands_parser(self, parser_subcommands: _ActionSubCommands) -> None:
        if _ENGINE_AVAILABLE:
            # `getitune find` arguments
            find_parser = ArgumentParser(formatter_class=CustomHelpFormatter)
            find_parser.add_argument(
                "--task",
                help="Value for filtering by task. Default is None, which shows all recipes.",
                type=Optional[TaskType],
            )
            find_parser.add_argument(
                "--pattern",
                help="This allows you to filter the model name of the recipe. \
                      For example, if you want to find all models that contain the word 'efficient', \
                      you can use '--pattern efficient'",
                type=Optional[str],
            )
            parser_subcommands.add_subcommand("find", find_parser, help="This shows the model provided by getitune.")

    def instantiate_classes(self, instantiate_engine: bool = True) -> None:
        """Instantiate the necessary classes based on the subcommand.

        This method checks if the subcommand is one of the engine subcommands.
        If it is, it instantiates the necessary classes such as config, datamodule, model, and engine.

        Args:
            instantiate_engine (bool, optional): Whether to instantiate the engine. Defaults to True.
        """
        if self.subcommand in self.engine_subcommands():
            # For num_classes update, Model and Metric are instantiated separately.

            model_config = self.config[self.subcommand].pop("model")

            # Instantiate the things that don't need to special handling
            self.config_init = self.parser.instantiate_classes(self.config)
            self.workspace = self.get_config_value(self.config_init, "workspace")
            self.datamodule = self.get_config_value(self.config_init, "data")

            # pass DataModule input size, mean and std to the model
            if self.datamodule.input_size is None:
                msg = (
                    "Input size is not specified in the datamodule. Ensure that the datamodule has a valid input size."
                )
                raise ValueError(msg)
            # Only pass what the datamodule knows; mean/std may be None when
            # normalization lives in the augmentation pipeline. Model defaults
            # fill in any missing values inside _configure_preprocessing_params.
            _dip: dict[str, Any] = {"input_size": self.datamodule.input_size}
            if self.datamodule.input_mean is not None:
                _dip["mean"] = self.datamodule.input_mean
            if self.datamodule.input_std is not None:
                _dip["std"] = self.datamodule.input_std
            _intensity_cfg = getattr(self.datamodule, "input_intensity_config", None)
            if _intensity_cfg is not None:
                _dip["intensity_config"] = asdict(_intensity_cfg)
            model_config.init_args["data_input_params"] = _dip

            # Instantiate the model and needed components
            self.model = self.instantiate_model(model_config=model_config)

            if instantiate_engine:
                self.engine = self.instantiate_engine()

    def instantiate_engine(self) -> LightningEngine:
        """Instantiate an Engine object with the specified parameters.

        Returns:
            An instance of the Engine class.
        """
        engine_kwargs = self.get_config_value(self.config_init, "engine")

        return LightningEngine(
            model=self.model,
            data=self.datamodule,
            work_dir=self.workspace.work_dir,
            **engine_kwargs,
        )

    def instantiate_model(self, model_config: Namespace) -> LightningModel:
        """Instantiate the model based on the subcommand.

        This method checks if the subcommand is one of the engine subcommands.
        If it is, it instantiates the model.

        Args:
            model_config (Namespace): The model configuration.

        Returns:
            tuple: The model and optimizer and scheduler.
        """
        from getitune.backend.lightning.models.base import LightningModel
        from getitune.utils.utils import can_pass_tile_config, get_model_cls_from_config, should_pass_label_info

        skip = set()

        # Update label_info
        model_cls = get_model_cls_from_config(model_config)

        if should_pass_label_info(model_cls) and not self.get_config_value(
            self.config_init,
            "disable_infer_num_classes",
            False,
        ):
            model_config.init_args.label_info = self.datamodule.label_info
            warning_msg = (
                "Automatically infer label_info from the given dataset. "
                "Then, giving it to the LightningModel.__init__() argument. "
                "If you don't want this behavior, please use `--disable-infer-num-classes` option."
            )
            warn(warning_msg, stacklevel=0)
            skip.add("label_info")

        # Update tile config due to adaptive tiling
        if can_pass_tile_config(model_cls):
            model_config.init_args.tile_config = self.datamodule.tile_config
            skip.add("tile_config")

        # NOTE: Workaround for jsonargparse cannot parse lambda default with unknown reasons
        optimizer_arg, scheduler_arg = model_config.init_args.get("optimizer"), model_config.init_args.get("scheduler")
        if isinstance(optimizer_arg, str) and optimizer_arg.endswith("<lambda>"):
            model_config.init_args.pop("optimizer")
        if isinstance(scheduler_arg, str) and scheduler_arg.endswith("<lambda>"):
            model_config.init_args.pop("scheduler")

        # Parses the LightningModel separately to update num_classes.
        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(LightningModel, "model", skip=skip, required=False, fail_untyped=False)
        model: LightningModel = model_parser.instantiate_classes(Namespace(model=model_config)).get("model")
        self.config_init[self.subcommand]["model"] = model

        # Update self.config with model
        self.config[self.subcommand].update(Namespace(model=model_config))

        return model

    def get_config_value(self, config: Namespace, key: str, default: Any = None) -> Any:  # noqa: ANN401
        """Retrieves the value of a configuration key from the given config object.

        Args:
            config (Namespace): The config object containing the configuration values.
            key (str): The key of the configuration value to retrieve.
            default (Any, optional): The default value to return if the key is not found. Defaults to None.

        Returns:
            Any: The value of the configuration key, or the default value if the key is not found.
                if the value is a Namespace, it is converted to a dictionary.
        """
        result = config.get(str(self.subcommand), config).get(key, default)
        return namespace_to_dict(result) if isinstance(result, Namespace) else result

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

    def prepare_subcommand_kwargs(self, subcommand: str) -> dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        return {
            k: v for k, v in self.config_init[subcommand].items() if k in self._subcommand_method_arguments[subcommand]
        }

    def save_config(self, work_dir: Path) -> None:
        """Save the configuration for the specified subcommand.

        Args:
            work_dir (Path): The working directory where the configuration file will be saved.

        The configuration is saved as a YAML file in the engine's working directory.
        """
        self.config[self.subcommand].pop("workspace", None)
        self.config[self.subcommand]["work_dir"] = str(self.workspace.work_dir.parent)
        # TODO(vinnamki): Revisit it after changing the optimizer and scheduler instantiating.
        cfg = deepcopy(self.config.get(str(self.subcommand), self.config))
        cfg.model.init_args.pop("optimizer")
        cfg.model.init_args.pop("scheduler")
        cfg.model.init_args.pop("label_info")
        cfg.model.init_args.pop("tile_config")

        self.get_subcommand_parser(self.subcommand).save(
            cfg=cfg,
            path=work_dir / "configs.yaml",
            overwrite=True,
            multifile=False,
            skip_check=True,
        )

        # if train -> Update `.latest` folder
        self.update_latest(work_dir=work_dir)

    def update_latest(self, work_dir: Path) -> None:
        """Update the latest cache directory with the latest configurations and checkpoint file.

        Args:
            work_dir (Path): The working directory where the configurations and checkpoint files are located.
        """
        latest_dir = work_dir.parent / ".latest"
        latest_dir.mkdir(exist_ok=True)
        cache_dir = latest_dir / self.subcommand
        if cache_dir.is_symlink() or cache_dir.exists():
            cache_dir.unlink()
        target = Path("..") / work_dir.relative_to(work_dir.parent)
        try:
            cache_dir.symlink_to(target)
        except OSError:
            # Symlink creation may fail on Windows without admin rights.
            # Fall back to copying the target directory.
            import shutil

            shutil.copytree(work_dir, cache_dir)

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
        self.console.print(f"[blue]{LIBRARY_LOGO}[/blue] ver.{__version__}", justify="center")
        if self.subcommand == "find":
            from getitune.utils import list_models

            list_models(print_table=True, **self.config[self.subcommand])
        elif self.subcommand in self.engine_subcommands():
            self.set_seed()
            self.instantiate_classes()
            fn_kwargs = self.prepare_subcommand_kwargs(self.subcommand)
            fn = getattr(self.engine, self.subcommand)
            try:
                outputs = fn(**fn_kwargs)
                self._print_results(outputs=outputs)
            except Exception:
                self.console.print_exception(width=self.console.width)
                raise
            self.save_config(work_dir=Path(self.engine.work_dir))
        else:
            msg = f"Unrecognized subcommand: {self.subcommand}"
            raise ValueError(msg)

    def _print_results(self, outputs: Any) -> None:  # noqa: ANN401
        if outputs is None:
            return
        if self.subcommand == "train" and isinstance(outputs, dict):
            # Print Metric like 'getitune test'
            from rich.table import Column, Table
            from torch import Tensor

            table_headers = ["Train metric", "Value"]
            columns = [Column(h, justify="center", style="magenta", width=self.console.width) for h in table_headers]
            columns[0].style = "cyan"
            table = Table(*columns)
            for metric, row in outputs.items():
                if isinstance(row, Tensor):
                    row = row.item() if row.numel() == 1 else row.tolist()  # noqa: PLW2901
                table.add_row(*[metric, f"{row}"])
            self.console.print(table)
        elif self.subcommand == "export":
            # Print output model path
            self.console.print(f"{self.subcommand} output: {outputs}")
        self.console.print(f"Work Directory: {self.engine.work_dir}")
