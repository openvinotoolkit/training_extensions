"""OTX CLI."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type, Union

from jsonargparse import (
    ActionConfigFile,
    Namespace,
    class_from_function,
    namespace_to_dict,
)

# Add the constructor to the YAML loader
from rich.console import Console

from otx.v2 import OTX_LOGO, __version__
from otx.v2.api.core import AutoRunner, BaseDataset, Engine
from otx.v2.api.utils.logger import get_logger
from otx.v2.cli.extensions import CLI_EXTENSIONS
from otx.v2.cli.utils.arg_parser import OTXArgumentParser, get_short_docstring
from otx.v2.cli.utils.help_formatter import pre_parse_arguments, render_guide
from otx.v2.cli.utils.workspace import Workspace

ArgsType = Optional[Union[list, dict, Namespace]]


class OTXCLIv2:
    """The main parser for the demo project."""

    def __init__(
        self,
        args: ArgsType = None,
        parser_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize a new instance of the CLI class.

        Args:
            args (ArgsType, optional): Command-line arguments to parse. Defaults to None.
            parser_kwargs (dict, optional): Additional keyword arguments to pass to the parser. Defaults to None.
        """
        self.console = Console()
        self.error: Optional[Exception] = None
        self.model_name: Optional[str] = None
        self.pre_args = {}
        self.auto_runner_class = AutoRunner

        # Checks to see if the user's command enables auto-configuration.
        self.pre_args = pre_parse_arguments()
        # To eliminate unnecessary output from print_config (To save the configuration file).
        if "print_config" not in self.pre_args:
            self.console.print(f"[blue]{OTX_LOGO}[/blue] ver.{__version__}", justify="center")
        else:
            logger = get_logger()
            logger.disabled = True

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
        self.parser = self.setup_parser(main_kwargs, subcommand_kwargs)

        # Main Parse Arguments
        self.config = self.parse_arguments(self.parser, args)

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

    def _setup_parser_kwargs(self, parser_kwargs: Optional[dict] = None) -> tuple:
        if parser_kwargs is None:
            parser_kwargs = {}
        subcommand_names = self.engine_subcommands().keys()
        main_kwargs = {k: v for k, v in parser_kwargs.items() if k not in subcommand_names}
        subparser_kwargs = {k: v for k, v in parser_kwargs.items() if k in subcommand_names}
        return main_kwargs, subparser_kwargs

    def init_parser(self, default_config_files: Optional[List[Optional[str]]] = None, **kwargs) -> OTXArgumentParser:
        """Initialize the argument parser for the OTX CLI.

        Args:
            default_config_files (Optional[List[Optional[str]]]): List of default configuration files.
            **kwargs: Additional keyword arguments to pass to the OTXArgumentParser constructor.

        Returns:
            OTXArgumentParser: The initialized argument parser.
        """
        parser = OTXArgumentParser(default_config_files=default_config_files, **kwargs)
        parser.add_argument(
            "-V",
            "--version",
            action="version",
            version=f"%(prog)s {__version__}",
            help="Display OTX version number.",
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
            help="Path to a configuration file in yaml format.",
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
        main_kwargs: dict,
        subparser_kwargs: dict,
    ) -> OTXArgumentParser:
        """Initialize and setup the parser, subcommands, and arguments."""
        parser = self.init_parser(**main_kwargs)
        self._subcommand_method_arguments: Dict[str, List[str]] = {}
        self.parser_subcommands = parser.add_subcommands()
        self._set_extension_subcommands_parser()
        self._set_engine_subcommands_parser(**subparser_kwargs)
        return parser

    @staticmethod
    def engine_subcommands() -> Dict[str, Set[str]]:
        """Return a dictionary of subcommands and their required arguments for the engine command.

        Returns:
            A dictionary where the keys are the subcommands and the values are sets of required arguments.
        """
        return {
            "train": {"model", "train_dataloader", "val_dataloader"},
            "validate": {"model", "val_dataloader"},
            "test": {"model", "test_dataloader"},
            "predict": {"model"},
            "export": {"model"},
        }

    def _set_engine_subcommands_parser(self, **kwargs) -> None:
        # the user might have passed a builder function
        self._engine_class = (
            self.framework_engine
            if isinstance(self.framework_engine, type)
            else class_from_function(self.framework_engine)
        )

        # register all subcommands in separate subcommand parsers under the main parser
        for subcommand in self.engine_subcommands():
            fn = getattr(self._engine_class, subcommand)
            # extract the first line description in the docstring for the subcommand help message
            description = get_short_docstring(fn)
            subparser_kwargs = kwargs.get(subcommand, {})
            subparser_kwargs.setdefault("description", description)
            subcommand_parser = self._prepare_subcommand_parser(self._engine_class, subcommand, **subparser_kwargs)
            self.parser_subcommands.add_subcommand(subcommand, subcommand_parser, help=description)

    def _prepare_subcommand_parser(self, klass: Type, subcommand: str, **kwargs) -> OTXArgumentParser:
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

    def _set_extension_subcommands_parser(self) -> None:
        for sub_command, functions in CLI_EXTENSIONS.items():
            add_parser_function = functions.get("add_parser", None)
            if add_parser_function is None:
                msg = f"The sub-parser function of {sub_command} was not found."
                raise NotImplementedError(msg)
            add_parser_function(self.parser_subcommands)

    def get_auto_runner(self) -> Optional[AutoRunner]:
        """Return an instance of AutoRunner class with the specified configuration parameters.

        If the user puts --checkpoint in the command and doesn't put --config,
        will use those configs as the default if they exist in the checkpoint folder location.
        """
        auto_runner = None
        if "checkpoint" in self.pre_args and self.pre_args.get("config", None) is None:
            checkpoint_path = self.pre_args.get("checkpoint", None)
            if checkpoint_path is not None:
                config_candidate = Path(checkpoint_path).parent / "configs.yaml"
                if config_candidate.exists():
                    self.pre_args["config"] = str(config_candidate)
                elif Path(checkpoint_path).exists():
                    msg = f"{config_candidate} not found. Please include --config."
                    raise FileNotFoundError(msg)
                else:
                    msg = f"{checkpoint_path} not found. Double-check your checkpoint file."
                    raise FileNotFoundError(msg)
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
        except Exception as e:
            self.error = e
        return auto_runner

    def get_model_class(self) -> tuple:
        """Return the model class and default configurations for the CLI.

        Returns:
            tuple: A tuple containing the model class and default configurations.
        """
        model_class = None
        default_configs = None
        if self.auto_runner is not None:
            self.auto_runner.build_framework_engine()
            framework_engine = self.auto_runner.engine
            default_configs = []
            # Find Model
            model_name = None
            if "model.name" in self.pre_args:
                model_name = self.pre_args["model.name"]
            elif hasattr(framework_engine, "config"):
                model_cfg = framework_engine.config.get("model", {})
                model_name = model_cfg.get("name", model_cfg.get("type", None))
            if model_name is None:
                msg = "The appropriate model was not found in config.."
                raise ValueError(msg)
            model = self.auto_runner.get_model(model=model_name)
            if model is None:
                msg = f"The model was not built: {model_name}"
                raise ValueError(msg)
            if model_name in self.auto_runner.config_list:
                default_configs.append(self.auto_runner.config_list[model_name])
            self.model_name = model_name
            # TODO: Need more flexible way for Model API
            default_configs.append(self.auto_runner.config_path)
            model_class = model.__class__
        return model_class, default_configs

    def parse_arguments(self, parser: OTXArgumentParser, args: ArgsType) -> Namespace:
        """Parse command line arguments using the provided parser.

        Args:
            parser (OTXArgumentParser): The argument parser to use.
            args (ArgsType): The command line arguments to parse.

        Returns:
            Namespace: The parsed arguments as a namespace object.
        """
        if isinstance(args, (dict, Namespace)):
            return parser.parse_object(args)
        return parser.parse_args(args)

    def instantiate_classes(self, subcommand: str) -> None:
        """Instantiate the necessary classes for running the command.

        Args:
            subcommand (str): The subcommand to be executed.

        Raises:
            ValueError: If the auto_runner is None or if the data configuration is not a dictionary or Namespace.
            TypeError: If the model configuration is not a dictionary or Namespace.

        Returns:
            None
        """
        if self.auto_runner is None:
            if self.error is not None:
                # Raise an existing raised exception only when the actual command is executed.
                raise self.error
            msg = "Couldn't run because it couldn't find a suitable task. Make sure you have enough commands entered."
            raise ValueError(
                msg,
            )
        self.config_init = self.parser.instantiate_classes(self.config)
        num_classes = None
        workspace_config = {}
        if subcommand not in ["predict", "export"]:
            data_cfg = self._pop(self.config_init, "data")
            if not isinstance(data_cfg, (dict, Namespace)):
                msg = "There is a problem with data configuration. Please check it again."
                raise TypeError(msg)
            self.data = self.data_class(**data_cfg)
            num_classes = self.data.num_classes
            workspace_config["data"] = {**data_cfg}

        model_cfg = self._pop(self.config_init, "model")
        if not isinstance(model_cfg, (dict, Namespace)):
            msg = "There is a problem with model configuration. Please check it again."
            raise TypeError(msg)
        self.model = self.auto_runner.get_model(model={**model_cfg}, num_classes=num_classes)
        # For prediction class
        if num_classes is not None and "num_classes" in model_cfg.get("head", {}):
            model_cfg["head"]["num_classes"] = num_classes
        workspace_config["model"] = {**model_cfg}

        config = self._pop(self.config_init, "config")
        if config is not None and len(config) > 0:
            config = str(config[0])
        work_dir = self._pop(self.config_init, "work_dir")
        if not isinstance(work_dir, str):
            # TODO: Need to fix properly.
            work_dir = None

        # Workspace
        self.workspace = Workspace(work_dir=work_dir, task=str(self.auto_runner.task.name).lower())
        self.engine = self.framework_engine(
            work_dir=str(self.workspace.work_dir),
            config=config,
        )
        self.workspace.add_config(workspace_config)

    def _pop(
        self,
        config: Namespace,
        key: str,
        default: Optional[Union[dict, str, Namespace]] = None,
    ) -> Optional[Union[dict, str, Namespace]]:
        return config.get(str(self.subcommand), config).pop(key, default)

    def run(self, subcommand: str) -> None:
        """Run the specified subcommand.

        Args:
            subcommand (str): The subcommand to run.

        Raises:
            NotImplementedError: If the specified subcommand is not implemented.
        """
        start_time = time.time()
        if subcommand in CLI_EXTENSIONS:
            config = namespace_to_dict(self.config[subcommand])
            extension_function = CLI_EXTENSIONS[subcommand].get("main", None)
            if extension_function is None:
                msg = f"The main function of {subcommand} is not implemented."
                raise NotImplementedError(msg)
            extension_function(**config)
        elif subcommand == "train":
            self.instantiate_classes(subcommand)
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
                },
            )
            # TODO: Cleanup for output
            # The configuration dump is saved next to the checkpoint file.
            model_base_dir = Path(results["checkpoint"]).parent
            self.workspace.dump_config(filename=str(model_base_dir / "configs.yaml"))
            self.console.print(f"[*] OTX Model Weight: {results['checkpoint']}")
            self.console.print(f"[*] OTX configuration used in the training: {model_base_dir / 'configs.yaml'!s}")

            # Latest dir update
            self.workspace.update_latest(model_base_dir)

        elif subcommand == "test":
            self.instantiate_classes(subcommand)
            test_dl_kwargs = self._prepare_dataloader_kwargs(subcommand, "test")
            subcommand_kwargs, left_kwargs = self._prepare_subcommand_kwargs(subcommand)
            results = self.engine.test(
                self.model,
                test_dataloader=self.data.test_dataloader(**test_dl_kwargs),
                **subcommand_kwargs,
            )
            # TODO: Cleanup for output
            self.console.print(results)
        elif subcommand == "predict":
            self.instantiate_classes(subcommand)
            subcommand_kwargs, left_kwargs = self._prepare_subcommand_kwargs(subcommand)
            results = self.engine.predict(model=self.model, **subcommand_kwargs)
            # TODO: Cleanup for output
            self.console.print(results)
        elif subcommand == "export":
            self.instantiate_classes(subcommand)
            subcommand_kwargs, left_kwargs = self._prepare_subcommand_kwargs(subcommand)
            results = self.engine.export(model=self.model, **subcommand_kwargs)
            # TODO: Cleanup for output
            self.console.print("[*] Model exporting ended successfully.")
        else:
            msg = f"{subcommand} is not implemented."
            raise NotImplementedError(msg)
        end_time = time.time()
        total_time = str(datetime.timedelta(seconds=end_time - start_time))
        if subcommand in self.engine_subcommands():
            self.console.print(f"[*] otx {subcommand} time elapsed: {total_time}")

    def _prepare_subcommand_kwargs(self, subcommand: str) -> Tuple[Dict, Dict]:
        config = namespace_to_dict(self.config_init[subcommand])
        subcommand_kwargs = {}
        left_kwargs = {}
        for k, v in config.items():
            if k in self._subcommand_method_arguments[subcommand]:
                subcommand_kwargs[k] = v
            else:
                left_kwargs[k] = v

        return subcommand_kwargs, left_kwargs

    def _prepare_dataloader_kwargs(self, subcommand: str, subset: str) -> dict:
        dl_kwargs = self.config_init[subcommand].pop(f"{subset}_dataloader", None)
        dl_kwargs.pop("self", None)
        dl_kwargs.pop("subset", None)
        dl_kwargs.pop("dataset", None)
        return dl_kwargs


def main() -> None:
    """Entry point for OTX CLI.

    This function is a single entry point for all OTX CLI related operations:
    """
    OTXCLIv2()


if __name__ == "__main__":
    main()
