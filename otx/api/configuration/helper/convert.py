"""This module contains the definition for the `convert` function within the configuration helper.

This function can be used to convert a OTX configuration object to a dictionary or yaml representation.
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from enum import Enum
from typing import Type, TypeVar

import yaml
from omegaconf import DictConfig, OmegaConf

from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.elements import (
    ConfigurableEnum,
    ParameterGroup,
    metadata_keys,
)

ConvertTypeVar = TypeVar("ConvertTypeVar", str, DictConfig, dict)


def serialize_metadata(metadata_dict: dict, enum_to_str: bool = True) -> dict:
    """This function converts Enums in the metadata_dict to their string representation.

    It is used when converting between yaml and python object representation of the configuration.

    Args:
        metadata_dict (dict): Dictionary containing the metadata to convert
        enum_to_str (bool): True to convert enums to their string representation, False to leave them as Enums
    """
    for key, value in metadata_dict.items():
        if isinstance(value, Enum):
            if enum_to_str:
                metadata_dict[key] = str(value)
        if key == metadata_keys.UI_RULES:
            metadata_dict[key] = value.to_dict(enum_to_str)
    return metadata_dict


def parameter_group_to_dict(
    parameter_group: ParameterGroup,
    enum_to_str: bool = False,
    values_only: bool = False,
) -> dict:
    """Converts an instance of a `ParameterGroup` configuration element to its dictionary representation.

    Args:
        parameter_group (ParameterGroup): ParameterGroup to convert to dictionary representation
        enum_to_str (bool): Set to True to convert any Enum fields in the configuration to
            their string representation.
        values_only (bool): True to keep only the parameter values, and remove all meta
            data from the output dictionary

    Returns:
        dict: Nested dictionary with keys and values corresponding to the configuration
            defined in the instance of `ParameterGroup` for which the `to_dict` method was
            called.
    """
    parameter_group.update_auto_hpo_states()
    attribute_names = [attribute.name for attribute in parameter_group.__attrs_attrs__]  # type: ignore
    # The __attrs_attrs__ attribute is added through the attrs package, mypy doesn't recognize it so we can ignore the
    # type error
    attribute_values = [getattr(parameter_group, attribute_name) for attribute_name in attribute_names]

    dictionary_representation = {}
    for name, value in zip(attribute_names, attribute_values):
        # Go through all simple attributes first, add them to the dictionary representation
        if isinstance(value, Enum) and enum_to_str:
            value = str(value)
        dictionary_representation[name] = value
    for group_name in parameter_group.groups:
        # Then, recursively add all parameter groups to the dictionary representation
        group = getattr(parameter_group, group_name)
        dictionary_representation.update({group_name: parameter_group_to_dict(group, enum_to_str, values_only)})
    for parameter_name in parameter_group.parameters:
        # Then, add all parameters for this group to the dictionary representation
        # For each parameter, construct a dict with its value and then update it with the metadata
        value = getattr(parameter_group, parameter_name)
        if isinstance(value, ConfigurableEnum) and enum_to_str:
            value = str(value)
        if values_only:
            parameter_dictionary = value
        else:
            parameter_dictionary = {"value": value}
            parameter_dictionary.update(serialize_metadata(parameter_group.get_metadata(parameter_name), enum_to_str))
        # Finally, add the parameter to the dictionary representation of the group
        dictionary_representation.update({parameter_name: parameter_dictionary})
    return dictionary_representation


def convert(
    config: ConfigurableParameters,
    target: Type[ConvertTypeVar],
    enum_to_str: bool = False,
    id_to_str: bool = False,
    values_only: bool = False,
) -> ConvertTypeVar:
    """Convert a configuration object to either a yaml string, a dictionary or an OmegaConf DictConfig object.

    Args:
        config (ConfigurableParameters): ConfigurableParameters object to convert
        target (Type[ConvertTypeVar]): target type to convert to. Options are [str, dict, DictConfig]
        enum_to_str (bool) : Boolean specifying whether to convert enums within the config
                        to their string representation. For conversion to yaml, enums
                        are automatically converted and this option is disregarded.
        id_to_str (bool): True to convert the id of the configurable parameters to a string
            representation, False to leave it as an ID object
        values_only (bool): True to keep only the parameter values, and remove all meta
            data from the target output

    Raises:
        ValueError: if an unsupported conversion target is supplied

    Returns:
        ConvertTypeVar: Result of the conversion, the configuration specified in `config` in the
            representation specified in `target`
    """
    if target == str:
        enum_to_str = True

    config_dict = parameter_group_to_dict(config, enum_to_str=enum_to_str, values_only=values_only)

    if id_to_str or target == str or target == DictConfig:
        config_id = config_dict.get("id", None)
        config_dict["id"] = str(config_id) if config_id is not None else None

    if target == str:
        result = yaml.dump(config_dict)
    elif target == dict:
        result = config_dict
    elif target == DictConfig:
        result = OmegaConf.create(config_dict)
    else:
        raise ValueError("Unsupported conversion target! Supported target types are [str, dict, DictConfig]")
    return result  # type: ignore
