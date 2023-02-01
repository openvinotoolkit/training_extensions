"""This module contains utility functions used within the configuration helper module."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import json
import os
from enum import Enum
from typing import Any, List, Tuple, Type, Union

import yaml
from omegaconf import DictConfig, OmegaConf

from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.enums.utils import get_enum_names
from otx.api.entities.id import ID

from .config_element_mapping import (
    GroupElementMapping,
    PrimitiveElementMapping,
    RuleElementMapping,
)
from .convert import convert


def _search_in_config_dict_inner(
    config_dict: dict,
    key_to_search: str,
    prior_keys: List[str] = None,
    results: List[Tuple[Any, List[str]]] = None,
) -> List[Tuple[Any, List[str]]]:
    """Helper function for the `search_in_config_dict` function defined below.

    Args:
        config_dict: dict to search in
        key_to_search: dict key to look for
        prior_keys: List of prior keys leading to the key_to_search.
        results: List of previously found results

    Returns:
        List of (value_at_key_to_search, key_path_to_key_to_search)
        tuples, representing each occurrence of key_to_search within
        config_dict
    """
    if prior_keys is None:
        prior_keys = []
    if results is None:
        results = []
    if isinstance(config_dict, List):
        dict_to_search_in = dict(zip(range(len(config_dict)), config_dict))
    else:
        dict_to_search_in = config_dict
    if not (issubclass(type(dict_to_search_in), dict) or isinstance(dict_to_search_in, DictConfig)):
        return results
    for key, value in dict_to_search_in.items():
        current_key_path = prior_keys + [key]
        if key == key_to_search:
            results.append((value, prior_keys))
        _search_in_config_dict_inner(value, key_to_search, current_key_path, results)
    return results


def search_in_config_dict(config_dict: dict, key_to_search: str) -> List[Tuple[Any, List[str]]]:
    """Recursively searches a config_dict for all instances of key_to_search and returns the key path to them.

    Args:
        config_dict: dict to search in
        key_to_search: dict key to look for

    Returns:
        List of (value_at_key_to_search, key_path_to_key_to_search)
        tuples, representing each occurrence of key_to_search within
        config_dict
    """
    return _search_in_config_dict_inner(config_dict, key_to_search=key_to_search)


def input_to_config_dict(input_config: Union[str, DictConfig, dict], check_config_type: bool = True) -> dict:
    """Validate input configuration.

    Takes an input_config which can be a string, filepath, dict or DictConfig and performs basic validation that it
    can be converted into a configuration.

    Args:
        input_config: String, filepath, dict or DictConfig describing a
            configuration
        check_config_type: True to check that the input has a proper
            `type` attribute in order to be converted into a
            ConfigurableParameters object. False to disable this check.
            Defaults to True.

    Returns:
        dictionary or DictConfig
    """
    valid_types = (
        get_enum_names(PrimitiveElementMapping)
        + get_enum_names(RuleElementMapping)
        + get_enum_names(GroupElementMapping)
    )

    if isinstance(input_config, str):
        if os.path.exists(input_config):
            with open(input_config, "r", encoding="UTF-8") as file:
                result = yaml.safe_load(file)
        else:
            result = yaml.safe_load(input_config)
    elif issubclass(type(input_config), dict):
        result = input_config
    elif isinstance(input_config, DictConfig):
        result = OmegaConf.to_container(input_config)
    else:
        raise ValueError('Invalid input_config type! Valid types are "str", "DictConfig", "dict".')

    if check_config_type:
        config_type = str(result.get("type", None))

        if config_type is None:
            raise ValueError(
                f"Input cannot be converted to a valid configuration. No "
                f"configuration type was found in the following input: {input_config}"
            )
        if config_type not in valid_types:
            raise ValueError(
                f"Invalid configuration element type: {config_type} found in input. "
                f"Unable to parse input. Supported configuration element types are:"
                f" {valid_types}"
            )
    return result


def deserialize_enum_value(value: Union[str, Enum], enum_type: Type[Enum]):
    """Deserializes a value to an instance of a certain Enum.

    This checks whether the `value` passed is already an instance of the target Enum, in which case this function just
    returns the input `value`. If value is a string, this function returns the corresponding instance of the Enum
    passed in `enum_type`.

    Args:
        value: value to deserialize
        enum_type: class (should be a subclass of Enum) that the name
            belongs to

    Returns:
        instance of `enum_type`.`value`
    """
    if isinstance(value, enum_type):
        instance = value
    elif isinstance(value, str):
        try:
            instance = enum_type[value.upper()]
        except KeyError:
            try:
                instance = enum_type(value)  # type: ignore
            except ValueError as error:
                raise KeyError(f"Value `{value}` cannot be converted to an instance of " f"`{enum_type}`") from error
    else:
        raise ValueError(
            f"Invalid input data type, {type(value)} cannot be converted to an instance " f"of {enum_type}."
        )
    return instance


def ids_to_strings(config_dict: dict) -> dict:
    """Converts ID's in the `config_dict` to their string representation.

    Args:
        config_dict: Dictionary in which to replace the ID's by strings

    Returns:
        Updated config_dict dictionary
    """
    for key, value in config_dict.items():
        if isinstance(value, ID):
            config_dict[key] = str(value)
    return config_dict


def config_to_bytes(config: ConfigurableParameters) -> bytes:
    """Converts ConfigurableParameters to bytes.

    Args:
        config: configurable parameters

    Retruns:
        JSON in bytes
    """
    config_dict = convert(config, dict, enum_to_str=True)
    return json.dumps(config_dict, indent=4).encode()
