import importlib
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import docstring_parser
import torch
from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    Namespace,
    class_from_function,
    register_unresolvable_import_paths,
)

register_unresolvable_import_paths(torch)

import yaml
from jsonargparse._loaders_dumpers import DefaultLoader

from otx.v2.api.core import AutoEngine, BaseDataset, Engine
from otx.v2.cli import cli_otx_logo


# Custom Constructor for Python tuples
def tuple_constructor(loader, node):
    if isinstance(node, yaml.SequenceNode):
        # Load the elements as a list
        elements = loader.construct_sequence(node)
        # Return the tuple
        return tuple(elements)
    return None


# Add the constructor to the YAML loader
DefaultLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)


def pre_parse_arguments():
    arguments = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            key = sys.argv[i][2:]
            value = None
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                value = sys.argv[i + 1]
                i += 1
            arguments[key] = value
        i += 1
    return arguments


def get_short_docstring(component: object) -> Optional[str]:
    """Gets the short description from the docstring.

    Args:
        component (object): The component to get the docstring from

    Returns:
        Optional[str]: The short description
    """
    if component.__doc__ is None:
        return None
    docstring = docstring_parser.parse(component.__doc__)
    return docstring.short_description


class OTXArgumentParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser for OTX."""

    def __init__(
        self,
        *args: Any,
        description: str = "OpenVINO Training-Extension command line tool",
        env_prefix: str = "otx",
        default_env: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, description=description, env_prefix=env_prefix, default_env=default_env, **kwargs)

    def add_core_class_args(
        self,
        api_class,
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
        instantiate: bool = False,
    ) -> List[str]:
        """Adds arguments from a class to a nested key of the parser.

        Args:
            api_class: A callable or any subclass.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
            required: Whether the argument group is required.

        Returns:
            A list with the names of the class arguments added.
        """
        if callable(api_class) and not isinstance(api_class, type):
            api_class = class_from_function(api_class)

        if isinstance(api_class, type):
            if subclass_mode:
                return self.add_subclass_arguments(api_class, nested_key, fail_untyped=False, required=required)
            return self.add_class_arguments(
                api_class,
                nested_key,
                fail_untyped=False,
                instantiate=instantiate,
                sub_configs=True,
            )
        raise NotImplementedError()

    def check_config(
        self,
        cfg: Namespace,
        skip_none: bool = True,
        skip_required: bool = True,
        branch: Optional[str] = None,
    ):
        # Skip This one for Flexible Configuration
        pass


ArgsType = Optional[Union[List[str], Dict[str, Any], Namespace]]


class OTXCLIv2:
    """The main parser for the demo project."""

    def __init__(
        self,
        args: ArgsType = None,
        parser_kwargs: Dict[str, Any] = {},
    ):
        cli_otx_logo()
        self.engine_defaults = {}
        self.pre_args = {}
        self.auto_engine_class = AutoEngine

        # Checks to see if the user's command enables auto-configuration.
        self.auto_engine = self.get_auto_engine()
        if self.auto_engine is not None:
            self.framework_engine = self.auto_engine.framework_engine
            self.data_class = self.auto_engine.dataset
            self.model_class = self.get_model_class()
        else:
            self.framework_engine = Engine
            self.data_class = BaseDataset
            self.model_class = None

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(parser_kwargs)
        self.setup_parser(True, main_kwargs, subparser_kwargs)

        # Main Parse Arguments
        self.parse_arguments(self.parser, args)

        self.subcommand = self.config["subcommand"]
        self.instantiate_classes()

        if self.subcommand is not None:
            self.run(self.subcommand)

    def _setup_parser_kwargs(self, parser_kwargs: Dict[str, Any] = {}) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        subcommand_names = self.subcommands().keys()
        main_kwargs = {k: v for k, v in parser_kwargs.items() if k not in subcommand_names}
        subparser_kwargs = {k: v for k, v in parser_kwargs.items() if k in subcommand_names}
        return main_kwargs, subparser_kwargs

    def init_parser(self, **kwargs: Any) -> OTXArgumentParser:
        """Method that instantiates the argument parser."""
        parser = OTXArgumentParser(**kwargs)
        # Same with Engine's __init__ argument
        parser.add_argument(
            "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in yaml format."
        )
        parser.add_argument(
            "-o",
            "--work_dir",
            help="Path to store logs and outputs related to the command.",
            type=str,
        )
        return parser

    def setup_parser(
        self, add_subcommands: bool, main_kwargs: Dict[str, Any], subparser_kwargs: Dict[str, Any]
    ) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        self.parser = self.init_parser(**main_kwargs)
        if add_subcommands:
            self._subcommand_method_arguments: Dict[str, List[str]] = {}
            self._add_subcommands(self.parser, **subparser_kwargs)
        else:
            self._add_arguments(self.parser)

    def add_core_arguments_to_parser(self, parser: OTXArgumentParser, subcommand: str) -> None:
        """Adds arguments from the core classes to the parser."""
        engine_defaults = {"engine." + k: v for k, v in self.engine_defaults.items()}
        parser.set_defaults(engine_defaults)

        if self.model_class is not None:
            parser.add_core_class_args(self.model_class, "model", subclass_mode=False)
        parser.add_argument(
            "--model.type",
            help="Enter the class name of model.",
            default=self.pre_args.get("model.type", str(self.model_class)),
        )

        parser.add_core_class_args(self.data_class, "data", subclass_mode=False)

        for sub_command_arg in self.subcommands()[subcommand]:
            if "_dataloader" in sub_command_arg:
                subset = sub_command_arg.split("_")[0]
                parser.add_core_class_args(self.data_class.subset_dataloader, sub_command_arg, subclass_mode=False)
                parser.set_defaults({f"{sub_command_arg}.subset": subset})
            elif sub_command_arg == "dataloader":
                # TODO: engine.predict()
                pass

    def _add_arguments(self, parser: OTXArgumentParser, subcommand: str) -> None:
        # default + core + custom arguments
        self.add_core_arguments_to_parser(parser, subcommand)

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        return {
            "train": {"model", "train_dataloader", "val_dataloader"},
            "validate": {"model", "val_dataloader"},
            "test": {"model", "test_dataloader"},
            "predict": {"model", "dataloader", "checkpoint"},
            "export": {"model", "checkpoint"},
        }

    def _add_subcommands(self, parser: OTXArgumentParser, **kwargs: Any) -> None:
        """Adds subcommands to the input parser."""
        self._subcommand_parsers: Dict[str, OTXArgumentParser] = {}
        parser_subcommands = parser.add_subcommands()
        # the user might have passed a builder function
        self._engine_class = (
            self.framework_engine
            if isinstance(self.framework_engine, type)
            else class_from_function(self.framework_engine)
        )

        # register all subcommands in separate subcommand parsers under the main parser
        for subcommand in self.subcommands():
            fn = getattr(self._engine_class, subcommand)
            # auto_fn = getattr(self._auto_engine_class, subcommand)
            # extract the first line description in the docstring for the subcommand help message
            description = get_short_docstring(fn)
            subparser_kwargs = kwargs.get(subcommand, {})
            subparser_kwargs.setdefault("description", description)
            subcommand_parser = self._prepare_subcommand_parser(self._engine_class, subcommand, **subparser_kwargs)
            self._subcommand_parsers[subcommand] = subcommand_parser
            parser_subcommands.add_subcommand(subcommand, subcommand_parser, help=description)

    def _prepare_subcommand_parser(self, klass: Type, subcommand: str, **kwargs: Any) -> OTXArgumentParser:
        parser = self.init_parser(**kwargs)
        self._add_arguments(parser, subcommand)
        # subcommand arguments
        skip: Set[Union[str, int]] = set(self.subcommands()[subcommand])
        added = parser.add_method_arguments(klass, subcommand, skip=skip)
        # need to save which arguments were added to pass them to the method later
        self._subcommand_method_arguments[subcommand] = added
        return parser

    def get_auto_engine(self):
        pre_args = pre_parse_arguments()
        try:
            temp_engine = self.auto_engine_class(
                framework=pre_args.get("framework", None),
                task=pre_args.get("data.task", pre_args.get("task", None)),
                train_type=pre_args.get("data.train_type", None),
                work_dir=pre_args.get("data.work_dir", None),  # FIXME
                train_data_roots=pre_args.get("data.train_data_roots", None),
                train_ann_files=pre_args.get("data.train_ann_files", None),
                val_data_roots=pre_args.get("data.val_data_roots", None),
                val_ann_files=pre_args.get("data.val_ann_files", None),
                test_data_roots=pre_args.get("data.test_data_roots", None),
                test_ann_files=pre_args.get("data.test_ann_files", None),
                unlabeled_data_roots=pre_args.get("data.unlabeled_data_roots", None),
                unlabeled_file_list=pre_args.get("data.unlabeled_file_list", None),
                data_format=pre_args.get("data.data_format", None),
                config=pre_args.get("config", None),
            )
            self.pre_args = pre_args
            return temp_engine
        except:
            return None

    def get_model_class(self):
        framework_engine = self.auto_engine.build_framework_engine()
        if hasattr(framework_engine, "registry"):
            registry = framework_engine.registry
        else:
            raise NotImplementedError()
        # Find Model Name
        model_name = None
        if "model.type" in self.pre_args:
            model_name = self.pre_args["model.type"]
        else:
            model_cfg = framework_engine.config.get("model", {})
            model_name = model_cfg.get("type", model_cfg.get("name", None))
        if model_name is None:
            raise ValueError("The appropriate model was not found in config..")
        model = registry.get(model_name)
        if model is None:
            # TODO: Using get_model
            pass
        return model

    def parse_arguments(self, parser: OTXArgumentParser, args: ArgsType) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        if isinstance(args, (dict, Namespace)):
            self.config = parser.parse_object(args)
        else:
            self.config = parser.parse_args(args)

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        data_cfg = self._get(self.config_init, "data")
        model_cfg = self._get(self.config_init, "model")

        # Build Dataset
        self.data = self.data_class(**data_cfg)

        # Build Model
        self.model = self.auto_engine.get_model(model={**model_cfg}, num_classes=self.data.num_classes)

        config = self._get(self.config_init, "config")
        config = config[0] if len(config) > 0 else None
        work_dir = self._get(self.config_init, "work_dir")
        self.engine = self.framework_engine(
            work_dir=work_dir,
            config=str(config),
        )

    def _get(self, config: Namespace, key: str, default: Optional[Any] = None) -> Any:
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(self.subcommand), config).get(key, default)

    def run(self, subcommand: str):
        """Runs the subcommand."""
        subcommand_kwargs = self._prepare_subcommand_kwargs(subcommand)
        if subcommand == "install":
            pass
        elif subcommand == "train":
            train_dl_kwargs = self._prepare_dataloader_kwargs(subcommand, "train")
            val_dl_kwargs = self._prepare_dataloader_kwargs(subcommand, "val")
            self.engine.train(
                self.model,
                train_dataloader=self.data.train_dataloader(**train_dl_kwargs),
                val_dataloader=self.data.val_dataloader(**val_dl_kwargs),
                **subcommand_kwargs,
            )
        elif subcommand == "test":
            self._prepare_dataloader_kwargs(subcommand, "test")
            self.engine.test(
                self.model, test_dataloader=self.data.test_dataloader(**train_dl_kwargs), **subcommand_kwargs
            )
        elif subcommand == "list":
            pass
        else:
            for key, val in self.config[subcommand].items():
                print(f"{key}: {val}")

    def _prepare_subcommand_kwargs(self, subcommand: str) -> Dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        subcommand_kwargs = {
            k: v for k, v in self.config_init[subcommand].items() if k in self._subcommand_method_arguments[subcommand]
        }
        return subcommand_kwargs

    def _prepare_dataloader_kwargs(self, subcommand: str, subset: str) -> Dict[str, Any]:
        dl_kwargs = self.config_init[subcommand].get(f"{subset}_dataloader", None)
        dl_kwargs.pop("self", None)
        dl_kwargs.pop("subset", None)
        dl_kwargs.pop("dataset", None)
        return dl_kwargs


def main():
    """Entry point for OTX CLI.

    This function is a single entry point for all OTX CLI related operations:
    """

    OTXCLIv2()


if __name__ == "__main__":
    main()
