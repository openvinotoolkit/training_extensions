"""OTX CLI."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import datetime
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from jsonargparse import (
    ActionConfigFile,
    Namespace,
    class_from_function,
    namespace_to_dict,
)

# Add the constructor to the YAML loader
from jsonargparse._loaders_dumpers import DefaultLoader
from rich.console import Console

from otx.v2 import OTX_LOGO, __version__
from otx.v2.api.core import AutoRunner, BaseDataset, Engine
from otx.v2.cli.utils.help_formatter import OTXHelpFormatter, render_guide

from .extensions import CLI_EXTENSIONS
from .utils.arg_parser import OTXArgumentParser, get_short_docstring, pre_parse_arguments, tuple_constructor
from .utils.workspace import Workspace

DefaultLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)


ArgsType = Optional[Union[List[str], Dict[str, Any], Namespace]]


class OTXCLIv2:
    """The main parser for the demo project."""

    def __init__(
        self,
        args: ArgsType = None,
        parser_kwargs: Dict[str, Any] = {},
    ):
        self.console = Console()
        self.console.print(f"[blue]{OTX_LOGO}[/blue] ver.{__version__}", justify="center")
        self.error = None
        self.model_name = None
        self.pre_args = {}
        self.auto_runner_class = AutoRunner

        # Checks to see if the user's command enables auto-configuration.
        self.pre_args = pre_parse_arguments()

        # Setting Auto-Runner
        self.auto_runner = self.get_auto_runner()
        self.model_class, self.default_config_files = self.get_model_class()
        if self.auto_runner is not None:
            self.framework_engine = self.auto_runner.framework_engine
            self.data_class = self.auto_runner.dataset_class
        else:
            self.framework_engine = Engine
            self.data_class = BaseDataset

        main_kwargs, subcommand_kwargs = self._setup_parser_kwargs(parser_kwargs)
        self.setup_parser(main_kwargs, subcommand_kwargs)

        # Main Parse Arguments
        self.parse_arguments(self.parser, args)

        self.subcommand = self.config["subcommand"]

        if self.subcommand is not None:
            try:
                self.run(self.subcommand)
            except Exception:
                # Print subcommand guide
                contents = render_guide(self.subcommand)
                for content in contents:
                    self.console.print(content)
                # Print TraceBack
                self.console.print_exception(width=self.console.width)

    def _setup_parser_kwargs(self, parser_kwargs: Dict[str, Any] = {}) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        subcommand_names = self.engine_subcommands().keys()
        main_kwargs = {k: v for k, v in parser_kwargs.items() if k not in subcommand_names}
        subparser_kwargs = {k: v for k, v in parser_kwargs.items() if k in subcommand_names}
        return main_kwargs, subparser_kwargs

    def init_parser(
        self, default_config_files: Optional[List[Optional[str]]] = None, **kwargs: Any
    ) -> OTXArgumentParser:
        """Method that instantiates the argument parser."""
        parser = OTXArgumentParser(default_config_files=default_config_files, **kwargs)
        parser.add_argument(
            "-V", "--version", action="version", version=f"%(prog)s {__version__}", help="Display OTX version number."
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            help="Verbose mode. This shows a configuration argument that allows for more specific overrides. Multiple -v options increase the verbosity. The maximum is 2.",
        )
        parser.add_argument(
            "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in yaml format."
        )
        parser.add_argument(
            "-o",
            "--work_dir",
            help="Path to store logs and outputs related to the command.",
            type=str,
        )
        parser.add_argument(
            "--framework",
            help="Select Framework: {mmpretrain, anomalib}.",
            type=str,
        )
        return parser

    def setup_parser(
        self,
        main_kwargs: Dict[str, Any],
        subparser_kwargs: Dict[str, Any],
    ) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        self.parser = self.init_parser(**main_kwargs)
        # Check -hh or --verbose
        self._check_verbose_help_format()
        self._subcommand_method_arguments: Dict[str, List[str]] = {}
        self.parser_subcommands = self.parser.add_subcommands()
        self._set_extension_subcommands_parser()
        self._set_engine_subcommands_parser(**subparser_kwargs)

    @staticmethod
    def engine_subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        return {
            "train": {"model", "train_dataloader", "val_dataloader"},
            "validate": {"model", "val_dataloader"},
            "test": {"model", "test_dataloader"},
            "predict": {"model"},
            "export": {"model"},
        }

    def _set_engine_subcommands_parser(self, **kwargs) -> None:
        """Adds subcommands to the input parser."""
        # the user might have passed a builder function
        self._engine_class = (
            self.framework_engine
            if isinstance(self.framework_engine, type)
            else class_from_function(self.framework_engine)
        )

        # register all subcommands in separate subcommand parsers under the main parser
        for subcommand in self.engine_subcommands().keys():
            fn = getattr(self._engine_class, subcommand)
            # extract the first line description in the docstring for the subcommand help message
            description = get_short_docstring(fn)
            subparser_kwargs = kwargs.get(subcommand, {})
            subparser_kwargs.setdefault("description", description)
            subcommand_parser = self._prepare_subcommand_parser(self._engine_class, subcommand, **subparser_kwargs)
            self.parser_subcommands.add_subcommand(subcommand, subcommand_parser, help=description)

    def _prepare_subcommand_parser(self, klass: Type, subcommand: str, **kwargs: Any) -> OTXArgumentParser:
        parser = self.init_parser(default_config_files=self.default_config_files, **kwargs)
        if self.model_class is not None:
            parser.add_core_class_args(self.model_class, "model", subclass_mode=False)

        parser.add_argument(
            "--model.name",
            help="Enter the name of model.",
            default=self.model_name,
        )

        if subcommand not in ("predict", "export"):
            parser.add_core_class_args(self.data_class, "data", subclass_mode=False)

        for sub_command_arg in self.engine_subcommands()[subcommand]:
            if "_dataloader" in sub_command_arg:
                subset = sub_command_arg.split("_")[0]
                parser.add_core_class_args(self.data_class.subset_dataloader, sub_command_arg, subclass_mode=False)
                parser.set_defaults({f"{sub_command_arg}.subset": subset})

        # subcommand arguments
        skip: Set[Union[str, int]] = set(self.engine_subcommands()[subcommand])
        added = parser.add_method_arguments(klass, subcommand, skip=skip, fail_untyped=False)
        # need to save which arguments were added to pass them to the method later
        self._subcommand_method_arguments[subcommand] = added
        return parser

    def _set_extension_subcommands_parser(self, **kwargs) -> None:
        for sub_command, functions in CLI_EXTENSIONS.items():
            add_parser_function = functions.get("add_parser", None)
            if add_parser_function is None:
                raise NotImplementedError(f"The sub-parser function of {sub_command} was not found.")
            else:
                add_parser_function(self.parser)

    def get_auto_runner(self) -> Optional[AutoRunner]:
        # If the user puts --checkpoint in the command and doesn't put --config,
        # will use those configs as the default if they exist in the checkpoint folder location.
        if "checkpoint" in self.pre_args and self.pre_args.get("config", None) is None:
            checkpoint_path = self.pre_args.get("checkpoint", None)
            if checkpoint_path is not None:
                config_candidate = Path(checkpoint_path).parent / "configs.yaml"
                if config_candidate.exists():
                    self.pre_args["config"] = str(config_candidate)
                elif Path(checkpoint_path).exists():
                    raise FileNotFoundError(f"{config_candidate} not found. Please include --config.")
                else:
                    raise FileNotFoundError(f"{checkpoint_path} not found. Double-check your checkpoint file.")
        try:
            data_task = self.pre_args.get("data.task", None)
            auto_runner = self.auto_runner_class(
                framework=self.pre_args.get("framework", None),
                task=self.pre_args.get("task", None) if data_task is None else data_task,
                train_type=self.pre_args.get("data.train_type", None),
                work_dir=self.pre_args.get("work_dir", None),  # FIXME
                train_data_roots=self.pre_args.get("data.train_data_roots", None),
                train_ann_files=self.pre_args.get("data.train_ann_files", None),
                val_data_roots=self.pre_args.get("data.val_data_roots", None),
                val_ann_files=self.pre_args.get("data.val_ann_files", None),
                test_data_roots=self.pre_args.get("data.test_data_roots", None),
                test_ann_files=self.pre_args.get("data.test_ann_files", None),
                unlabeled_data_roots=self.pre_args.get("data.unlabeled_data_roots", None),
                unlabeled_file_list=self.pre_args.get("data.unlabeled_file_list", None),
                data_format=self.pre_args.get("data.data_format", None),
                config=self.pre_args.get("config", None),
            )
            return auto_runner
        except Exception as e:
            self.error = e
            return None

    def get_model_class(self) -> Tuple[Optional[Any], Optional[List[Optional[str]]]]:
        model_class = None
        default_configs = None
        if self.auto_runner is not None:
            self.auto_runner.build_framework_engine()
            framework_engine = self.auto_runner.engine
            default_configs = [self.auto_runner.config_path]
            # Find Model
            model_name = None
            if "model.name" in self.pre_args:
                model_name = self.pre_args["model.name"]
            elif hasattr(framework_engine, "config"):
                model_cfg = framework_engine.config.get("model", {})
                model_name = model_cfg.get("name", model_cfg.get("type", None))
            if model_name is None:
                raise ValueError("The appropriate model was not found in config..")
            model = self.auto_runner.get_model(model=model_name)
            if model is None:
                raise ValueError()
            if model_name in self.auto_runner.config_list:
                default_configs.append(self.auto_runner.config_list[model_name])
            self.model_name = model_name
            # TODO: Need more flexible way for Model API
            model_class = model.__class__
        return model_class, default_configs

    def parse_arguments(self, parser: OTXArgumentParser, args: ArgsType) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        if isinstance(args, (dict, Namespace)):
            self.config = parser.parse_object(args)
        else:
            self.config = parser.parse_args(args)

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        if self.auto_runner is None:
            if self.error is not None:
                # Raise an existing raised exception only when the actual command is executed.
                raise self.error
            raise ValueError(
                "Couldn't run because it couldn't find a suitable task. Make sure you have enough commands entered."
            )
        self.config_init = self.parser.instantiate_classes(self.config)
        data_cfg = self._pop(self.config_init, "data")
        model_cfg = self._pop(self.config_init, "model")

        # Build Dataset
        self.data = self.data_class(**data_cfg)
        self.model = self.auto_runner.get_model(model={**model_cfg}, num_classes=self.data.num_classes)
        # For prediction class
        if hasattr(model_cfg.get("head", None), "num_classes"):
            model_cfg["head"]["num_classes"] = self.data.num_classes

        config = self._pop(self.config_init, "config")
        if config is not None and len(config) > 0:
            config = str(config[0])
        work_dir = self._pop(self.config_init, "work_dir")

        # Workspace
        self.workspace = Workspace(work_dir=work_dir, task=str(self.auto_runner.task.name).lower())
        self.engine = self.framework_engine(
            work_dir=str(self.workspace.work_dir),
            config=config,
        )
        self.workspace.add_config({"data": {**data_cfg}, "model": {**model_cfg}})

    def _get(self, config: Namespace, key: str, default: Optional[Any] = None) -> Any:
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(self.subcommand), config).get(key, default)

    def _pop(self, config: Namespace, key: str, default: Optional[Any] = None) -> Any:
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(self.subcommand), config).pop(key, default)

    def run(self, subcommand: str):
        """Runs the subcommand."""
        start_time = time.time()
        if subcommand in CLI_EXTENSIONS:
            config = namespace_to_dict(self.config[subcommand])
            extension_function = CLI_EXTENSIONS[subcommand].get("main", None)
            if extension_function is None:
                raise NotImplementedError(f"The main function of {subcommand} is not implemented.")
            extension_function(**config)
        elif subcommand == "train":
            self.instantiate_classes()
            # Prepare Dataloader kwargs
            train_dl_kwargs = self._prepare_dataloader_kwargs(subcommand, "train")
            val_dl_kwargs = self._prepare_dataloader_kwargs(subcommand, "val")
            # Prepare subcommand kwargs
            subcommand_kwargs, left_kwargs = self._prepare_subcommand_kwargs(subcommand)
            results = self.engine.train(
                model=self.model,
                train_dataloader=self.data.train_dataloader(**train_dl_kwargs),
                val_dataloader=self.data.val_dataloader(**val_dl_kwargs),
                **subcommand_kwargs,
            )
            self.workspace.add_config(
                {
                    "train_dataloader": {**train_dl_kwargs},
                    "val_dataloader": {**val_dl_kwargs},
                    **subcommand_kwargs,
                    **left_kwargs,
                }
            )
            # TODO: Cleanup for output
            # The configuration dump is saved next to the checkpoint file.
            model_base_dir = Path(results["checkpoint"]).parent
            self.workspace.dump_config(filename=str(model_base_dir / "configs.yaml"))
            self.console.print(f"[*] OTX Model Weight: {results['checkpoint']}")
            self.console.print(f"[*] OTX configuration used in the training: {str(model_base_dir / 'configs.yaml')}")

            # Latest dir update
            self.workspace.update_latest(model_base_dir)

        elif subcommand == "test":
            self.instantiate_classes()
            test_dl_kwargs = self._prepare_dataloader_kwargs(subcommand, "test")
            subcommand_kwargs, left_kwargs = self._prepare_subcommand_kwargs(subcommand)
            results = self.engine.test(
                self.model, test_dataloader=self.data.test_dataloader(**test_dl_kwargs), **subcommand_kwargs
            )
            # TODO: Cleanup for output
            self.console.print(results)
        elif subcommand == "predict":
            self.instantiate_classes()
            subcommand_kwargs, left_kwargs = self._prepare_subcommand_kwargs(subcommand)
            results = self.engine.predict(model=self.model, **subcommand_kwargs)
            # TODO: Cleanup for output
            self.console.print(results)
        elif subcommand == "export":
            self.instantiate_classes()
            subcommand_kwargs, left_kwargs = self._prepare_subcommand_kwargs(subcommand)
            results = self.engine.export(model=self.model, **subcommand_kwargs)
            # TODO: Cleanup for output
            self.console.print("[*] Model exporting ended successfully.")
        else:
            for key, val in self.config[subcommand].items():
                self.console.print(f"{key}: {val}")
        end_time = time.time()
        total_time = str(datetime.timedelta(seconds=end_time - start_time))
        if subcommand in self.engine_subcommands():
            self.console.print(f"[*] otx {subcommand} time elapsed: {total_time}")

    def _prepare_subcommand_kwargs(self, subcommand: str) -> Tuple[Dict, Dict]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        config = namespace_to_dict(self.config_init[subcommand])
        subcommand_kwargs = {}
        left_kwargs = {}
        for k, v in config.items():
            if k in self._subcommand_method_arguments[subcommand]:
                subcommand_kwargs[k] = v
            else:
                left_kwargs[k] = v

        return subcommand_kwargs, left_kwargs

    def _prepare_dataloader_kwargs(self, subcommand: str, subset: str) -> Dict[str, Any]:
        dl_kwargs = self.config_init[subcommand].pop(f"{subset}_dataloader", None)
        dl_kwargs.pop("self", None)
        dl_kwargs.pop("subset", None)
        dl_kwargs.pop("dataset", None)
        return dl_kwargs

    def _check_verbose_help_format(self) -> None:
        subcommand = self.pre_args.get("subcommand", None)
        if issubclass(self.parser.formatter_class, OTXHelpFormatter):
            # TODO: how to use verbose count value
            if subcommand not in self.engine_subcommands():
                pass
            elif "v" in self.pre_args:
                sys.argv.append("--help")
                self.parser.formatter_class.verbose_level = 1
                self.parser.formatter_class.subcommand = subcommand
            elif "vv" in self.pre_args:
                sys.argv.append("--help")
                self.parser.formatter_class.verbose_level = 2
                self.parser.formatter_class.subcommand = subcommand
            elif subcommand is not None:
                # -v Applies only to subcommands provided by Engine.
                self.parser.formatter_class.verbose_level = 0
                self.parser.formatter_class.subcommand = subcommand
        else:
            self.console.print("The current Help Formatter does not support the verbose help format.")


def main() -> None:
    """Entry point for OTX CLI.

    This function is a single entry point for all OTX CLI related operations:
    """

    OTXCLIv2()


if __name__ == "__main__":
    main()
