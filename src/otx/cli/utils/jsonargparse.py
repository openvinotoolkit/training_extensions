"""Functions related to jsonargparse."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TypeVar

import docstring_parser
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace, dict_to_namespace, namespace_to_dict


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
            cfg.__dict__.update(overrides)
        if cfg.get(dest) is None:
            cfg[dest] = []
        cfg[dest].append(cfg_path)


@contextmanager
def patch_update_configs() -> Iterator[None]:
    """Patch the update and apply_config methods of the given namespace and action_config_file objects."""
    original_update = Namespace.update
    original_apply_config = ActionConfigFile.apply_config

    try:
        Namespace.update = update
        ActionConfigFile.apply_config = apply_config
        yield
    finally:
        Namespace.update = original_update
        ActionConfigFile.apply_config = original_apply_config


def get_configuration(config_path: str | Path) -> dict:
    """Get the configuration from the given path.

    Args:
        config_path (str | Path): The path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    from otx.cli.cli import OTXCLI
    with patch_update_configs():
        parser = OTXCLI.engine_subcommand_parser()
        args = parser.parse_args(args=["--config", str(config_path)], _skip_check=True)
    return namespace_to_dict(args)
