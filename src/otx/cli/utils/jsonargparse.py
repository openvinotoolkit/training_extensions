"""Functions related to jsonargparse."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TypeVar, Union

import docstring_parser
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace, dict_to_namespace, namespace_to_dict

from otx.core.types import PathLike

logger = logging.getLogger()


def get_short_docstring(component: TypeVar) -> str:
    """Get the short description from the docstring.

    Args:
        component (TypeVar): The component to get the docstring from

    Returns:
        str: The short description
    """
    if component.__doc__ is None:
        return ""
    docstring = docstring_parser.parse(component.__doc__)
    return docstring.short_description


def flatten_dict(config: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary into a single-level dictionary.

    Args:
        d (dict): The dictionary to be flattened.
        parent_key (str): The parent key to be used for nested keys.
        sep (str): The separator to be used between parent and child keys.

    Returns:
        dict: The flattened dictionary.

    """
    items: list = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# [FIXME]: Overriding Namespce.update to match mmengine.Config (DictConfig | dict)
# and prevent int, float types from being converted to str
# https://github.com/omni-us/jsonargparse/issues/236
def update(
    self: Namespace,
    value: Any,  # noqa: ANN401
    key: str | None = None,
    only_unset: bool = False,
) -> Namespace:
    """Sets or replaces all items from the given nested namespace.

    Args:
        value: A namespace to update multiple values or other type to set in a single key.
        key: Branch key where to set the value. Required if value is not namespace.
        only_unset: Whether to only set the value if not set in namespace.
    """
    is_value_dict = False
    if isinstance(value, dict):
        # Dict -> Nested Namespace for overriding
        is_value_dict = True
        value = dict_to_namespace(value)
    if not isinstance(value, (Namespace, dict)):
        if not key:
            msg = "Key is required if value not a Namespace."
            raise KeyError(msg)
        if not only_unset or key not in self:
            if key not in self or value is not None:
                if isinstance(value, str) and (value.isnumeric() or value in ("True", "False")):
                    value = ast.literal_eval(value)
                self[key] = value
            elif value is None:
                del self[key]
    else:
        prefix = key + "." if key else ""
        for _key, val in value.items():
            if not only_unset or prefix + _key not in self:
                self.update(val, prefix + _key)
    if is_value_dict and key is not None:
        # Dict or Namespace -> Dict
        self[key] = dict_to_namespace(self[key]).as_dict()
    return self


# To provide overriding of the Config file
def apply_config(self: ActionConfigFile, parser: ArgumentParser, cfg: Namespace, dest: str, value: str) -> None:  # noqa: ARG001
    """Applies the configuration to the parser.

    Args:
        parser: The parser object.
        cfg: The configuration object.
        dest: The destination attribute.
        value: The value to be applied.

    Returns:
        None
    """
    from jsonargparse._actions import _ActionSubCommands, previous_config_context
    from jsonargparse._link_arguments import skip_apply_links
    from jsonargparse._loaders_dumpers import get_loader_exceptions, load_value
    from jsonargparse._optionals import get_config_read_mode

    with _ActionSubCommands.not_single_subcommand(), previous_config_context(cfg), skip_apply_links():
        kwargs = {"env": False, "defaults": False, "_skip_check": True, "_fail_no_subcommand": False}
        try:
            cfg_path: Path | None = Path(value, mode=get_config_read_mode())
        except TypeError:
            try:
                if isinstance(load_value(value), str):
                    raise
                cfg_path = None
                cfg_file = parser.parse_string(value, **kwargs)
            except (TypeError, *get_loader_exceptions()) as ex_str:
                msg = f'Parser key "{dest}": {ex_str}'
                raise TypeError(msg) from ex_str
        else:
            cfg_file = parser.parse_path(value, **kwargs)
        cfg_merged = parser.merge_config(cfg_file, cfg)
        cfg.__dict__.update(cfg_merged.__dict__)
        overrides = cfg.__dict__.pop("overrides", None)
        if overrides is not None:
            apply_override(cfg, overrides)
            cfg.update(overrides)
        if cfg.get(dest) is None:
            cfg[dest] = []
        cfg[dest].append(cfg_path)


def namespace_override(
    configs: Namespace,
    key: str,
    overrides: Namespace,
    convert_dict_to_namespace: bool = True,
) -> None:
    """Overrides the nested namespace type in the given configs with the provided overrides.

    Args:
        configs (Namespace): The configuration object containing the key.
        key (str): key of the configs want to override.
        overrides (Namespace): The configuration object to override the existing ones.
        convert_dict_to_namespace (bool): Whether to convert the dictionary to Namespace. Defaults to True.
    """
    for sub_key, sub_value in overrides.items():
        if isinstance(sub_value, list) and all(isinstance(sv, dict) for sv in sub_value):
            # only enable list of dictionary items
            list_override(
                configs=configs[key],
                key=sub_key,
                overrides=sub_value,
                convert_dict_to_namespace=convert_dict_to_namespace,
            )
        else:
            configs[key].update(sub_value, sub_key)


def list_override(configs: Namespace, key: str, overrides: list, convert_dict_to_namespace: bool = True) -> None:
    """Overrides the nested list type in the given configs with the provided override_list.

    Args:
        configs (Namespace): The configuration object containing the key.
        key (str): key of the configs want to override.
        overrides (list): The list of dictionary item to override the existing ones.
        convert_dict_to_namespace (bool): Whether to convert the dictionary to Namespace. Defaults to True.

    Example:
        >>> configs = [
        ...     ...
        ...     Namespace(
        ...         class_path='lightning.pytorch.callbacks.EarlyStopping',
        ...         init_args=Namespace(patience=10, ...),
        ...     ),
        ...     ...
        ... ]
        >>> override_callbacks = [
        ...     ...
        ...     {
        ...         'class_path': 'lightning.pytorch.callbacks.EarlyStopping',
        ...         'init_args': {'patience': 3},
        ...     },
        ...     ...
        ... ]
        >>> list_override(configs=configs, key="callbacks", overrides=override_callbacks)
        >>> configs = [
        ...     ...
        ...     Namespace(
        ...         class_path='lightning.pytorch.callbacks.EarlyStopping',
        ...         init_args=Namespace(patience=3, ...),
        ...     ),
        ...     ...
        ... ]
        >>> append_callbacks = [
        ...     {
        ...         'class_path': 'new_callbacks',
        ...     },
        ... ]
        >>> list_override(configs=configs, key="callbacks", overrides=append_callbacks)
        >>> configs = [
        ...     ...
        ...     Namespace(class_path='new_callbacks'),
        ... ]
        >>> append_callbacks_as_dict = [
        ...     {
        ...         'class_path': 'new_callbacks1',
        ...     },
        ... ]
        >>> list_override(
        ...     configs=configs, key="callbacks", overrides=append_callbacks_as_dict, convert_dict_to_namespace=False
        ... )
        >>> configs = [
        ...     ...
        ...     {'class_path': 'new_callbacks1'},
        ... ]
    """
    if key not in configs or configs[key] is None:
        return
    for target in overrides:
        class_path = target.get("class_path", None)
        if class_path is None:
            msg = "class_path is required in the override list."
            raise ValueError(msg)

        item = next((item for item in configs[key] if item["class_path"] == class_path), None)
        if item is not None:
            Namespace(item).update(target)
        else:
            converted_target = dict_to_namespace(target) if convert_dict_to_namespace else target
            configs[key].append(converted_target)


def apply_override(cfg: Namespace, overrides: Namespace) -> None:
    """Overrides the provided overrides in the given configs.

    Args:
        configs (Namespace): The configuration object containing the key.
        overrides (Namespace): The configuration object to override the existing ones.
    """
    # replace the config with the overrides for keys in reset list
    reset = overrides.pop("reset", [])
    if isinstance(reset, str):
        reset = [reset]
    for key in reset:
        if key in overrides:
            # callbacks, logger -> update to namespace
            # rest -> use dict as is
            cfg[key] = (
                [dict_to_namespace(o) for o in overrides.pop(key)]
                if key in ("callbacks", "logger")
                else overrides.pop(key)
            )

    # This is a feature to handle the callbacks, logger, and data override for user-convinience
    list_override(configs=cfg, key="callbacks", overrides=overrides.pop("callbacks", []))
    list_override(configs=cfg, key="logger", overrides=overrides.pop("logger", []))
    namespace_override(
        configs=cfg,
        key="data",
        overrides=overrides.pop("data", Namespace()),
        convert_dict_to_namespace=False,
    )


# [FIXME] harimkang: have to see if there's a better way to do it. (For now, Added 2 lines to existing function)
# The thing called `overrides` is only available in OTXCLI via `apply_config`.
# Currently, default_config_files in jsonargparse is loading the default config file without using the ActionConfigFile,
# and it's not updating the overrides properly in the process.
# So this function patches to allow configs to come in via `default_config_files` with `overrides` applied.
def get_defaults_with_overrides(self: ArgumentParser, skip_check: bool = False) -> Namespace:
    """Returns a namespace with all default values.

    Args:
        skip_check: Whether to skip check if configuration is valid.

    Returns:
        An object with all default values as attributes.
    """
    import argparse

    from jsonargparse._actions import _ActionPrintConfig, filter_default_actions
    from jsonargparse._common import parser_context
    from jsonargparse._namespace import recreate_branches
    from jsonargparse._parameter_resolvers import UnknownDefault
    from jsonargparse._typehints import ActionTypeHint
    from jsonargparse._util import argument_error, change_to_path_dir

    cfg = Namespace()
    for action in filter_default_actions(self._actions):
        if (
            action.default != argparse.SUPPRESS
            and action.dest != argparse.SUPPRESS
            and not isinstance(action.default, UnknownDefault)
        ):
            cfg[action.dest] = recreate_branches(action.default)

    self._logger.debug("Loaded parser defaults: %s", cfg)

    default_config_files = self._get_default_config_files()
    for key, default_config_file in default_config_files:
        with change_to_path_dir(default_config_file), parser_context(parent_parser=self):
            cfg_file = self._load_config_parser_mode(default_config_file.get_content(), key=key)
            cfg = self.merge_config(cfg_file, cfg)
            overrides = cfg.__dict__.pop("overrides", {})
            apply_override(cfg, overrides)
            if overrides is not None:
                cfg.update(overrides)
            try:
                with _ActionPrintConfig.skip_print_config():
                    cfg = self._parse_common(
                        cfg=cfg,
                        env=False,
                        defaults=False,
                        with_meta=None,
                        skip_check=skip_check,
                        skip_required=True,
                    )
            except (TypeError, KeyError, argparse.ArgumentError) as ex:
                msg = f'Problem in default config file "{default_config_file}": {ex.args[0]}'
                raise argument_error(msg) from ex
        meta = cfg.get("__default_config__")
        if isinstance(meta, list):
            meta.append(default_config_file)
        elif isinstance(meta, Path):
            cfg["__default_config__"] = [meta, default_config_file]
        else:
            cfg["__default_config__"] = default_config_file
        self._logger.debug("Parsed default configuration from path: %s", default_config_file)

    ActionTypeHint.add_sub_defaults(self, cfg)

    return cfg


# Workaround for https://github.com/omni-us/jsonargparse/issues/456
def add_list_type_arguments(
    parser: ArgumentParser,
    baseclass: tuple[type, ...],
    nested_key: str,
    skip: set[str] | None = None,
) -> None:
    """Add list type arguments to the given ArgumentParser.

    From python >= 3.11, add_subclass_arguments no longer allows adding arguments of the form list[Class].
    Modify it to bypass class checking, allowing you to use the list argument.
    Copy from jsonargparse._signatures.SignatureArguments.add_subclass_arguments.

    Args:
        parser (ArgumentParser): The ArgumentParser to add the arguments to.
        baseclass (tuple[type, ...]): A tuple of base classes for the subclasses.
        nested_key (str): The nested key for the arguments.
        skip (set[str] | None, optional): A set of arguments to skip. Defaults to None.
    """
    from argparse import SUPPRESS

    from jsonargparse._parameter_resolvers import ParamData
    from jsonargparse._util import get_import_path, iter_to_set_str

    group = parser._create_group_if_requested(  # noqa: SLF001
        baseclass,
        nested_key,
        True,
        None,
        config_load=False,
        required=False,
        instantiate=False,
    )
    added_args: list[str] = []
    if skip is not None:
        skip = {f"{nested_key}.init_args." + s for s in skip}
    param = ParamData(name=nested_key, annotation=Union[baseclass], component=baseclass)
    str_baseclass = iter_to_set_str(get_import_path(x) for x in baseclass)
    kwargs = {
        "metavar": "CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE",
        "help": (
            f"One or more arguments specifying 'class_path' and 'init_args' for any subclass of {str_baseclass}s."
        ),
    }
    kwargs["default"] = SUPPRESS
    parser._add_signature_parameter(  # noqa: SLF001
        group,
        None,
        param,
        added_args,
        skip,
        sub_configs=True,
        instantiate=False,
        fail_untyped=False,
    )


@contextmanager
def patch_update_configs() -> Iterator[None]:
    """Patch the update and apply_config methods of the given namespace and action_config_file objects."""
    original_update = Namespace.update
    original_apply_config = ActionConfigFile.apply_config
    original_get_defaults = ArgumentParser.get_defaults

    try:
        Namespace.update = update
        ActionConfigFile.apply_config = apply_config
        ArgumentParser.get_defaults = get_defaults_with_overrides
        yield
    finally:
        Namespace.update = original_update
        ActionConfigFile.apply_config = original_apply_config
        ArgumentParser.get_defaults = original_get_defaults


def get_configuration(config_path: str | Path, subcommand: str = "train", **kwargs) -> dict:
    """Get the configuration from the given path.

    Args:
        config_path (str | Path): The path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    from otx.cli.cli import OTXCLI

    with patch_update_configs():
        parser, _ = OTXCLI.engine_subcommand_parser(subcommand=subcommand)
        if kwargs:
            parser.set_defaults(**kwargs)

        args = parser.parse_args(args=["--config", str(config_path)], _skip_check=True)

    config = namespace_to_dict(args)
    logger.info(f"{config_path} is loaded.")

    # Remove unnecessary cli arguments for API usage
    cli_args = [
        "verbose",
        "data_root",
        "task",
        "seed",
        "callback_monitor",
        "resume",
        "disable_infer_num_classes",
        "workspace",
    ]
    logger.warning(f"The corresponding keys in config are not used.: {cli_args}")
    for arg in cli_args:
        config.pop(arg, None)
    return config


def get_instantiated_classes(
    config: PathLike,
    work_dir: PathLike | None,
    data_root: PathLike | None,
    **kwargs,
) -> tuple[dict, dict]:
    """Get the instantiated classes for training.

    Args:
        config (PathLike): Path to the configuration file.
        work_dir (PathLike): Path to the working directory.
        data_root (PathLike): Path to the data root directory.

    Returns:
        dict: The instantiated classes for training.
    """
    from otx.cli import OTXCLI

    cli_args = [
        "train",
        "--config",
        str(config),
        "--workspace.use_sub_dir",
        "false",
    ]
    if work_dir is not None:
        cli_args.extend(["--work_dir", str(work_dir)])
    if data_root is not None:
        cli_args.extend(["--data_root", str(data_root)])
    for key, value in kwargs.items():
        cli_args.extend([f"--{key}", str(value)])
    otx_cli = OTXCLI(
        args=cli_args,
        run=False,
    )

    otx_cli.set_seed()
    otx_cli.instantiate_classes(instantiate_engine=False)
    instantiated_config = namespace_to_dict(otx_cli.config_init["train"])

    return instantiated_config, otx_cli.prepare_subcommand_kwargs("train")
