"""This module contains the definition for the `create` function within the configuration helper.

This function can be used to create a OTX configuration object from a dictionary or yaml representation.
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from __future__ import annotations

import copy
from enum import Enum
from typing import Dict, List, TypeVar, Union

import attr
from omegaconf import DictConfig, OmegaConf

from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.elements import (
    ConfigurableEnum,
    ParameterGroup,
    metadata_keys,
)
from otx.api.configuration.enums import AutoHPOState, ModelLifecycle
from otx.api.configuration.enums.config_element_type import (
    ConfigElementType,
    ElementCategory,
)
from otx.api.configuration.enums.utils import get_enum_names
from otx.api.configuration.ui_rules.rules import NullUIRules, Rule, UIRules

from .config_element_mapping import (
    GroupElementMapping,
    PrimitiveElementMapping,
    RuleElementMapping,
)
from .utils import deserialize_enum_value, input_to_config_dict

ParameterGroupTypeVar = TypeVar("ParameterGroupTypeVar", bound=ParameterGroup)
ExposureTypeVar = TypeVar("ExposureTypeVar", UIRules, Rule)

METADATA_ENUMS = {
    metadata_keys.AFFECTS_OUTCOME_OF: ModelLifecycle,
    metadata_keys.AUTO_HPO_STATE: AutoHPOState,
}


def construct_attrib_from_dict(dict_object: Union[dict, DictConfig]) -> ExposureTypeVar:
    """Constructs a ui exposure element from an input dictionary or DictConfig.

    Elements are mapped according to the 'type' field in the input dict.

    Args:
        dict_object (Union[dict, DictConfig]): Dictionary containing the arguments for the element constructor.

    Returns:
        ExposureTypeVar: Rule or UIRules element, constructed according to the input dict_object.
    """
    value_dict = dict(dict_object)
    object_type = str(value_dict.pop(metadata_keys.TYPE))
    if object_type in get_enum_names(RuleElementMapping):
        mapping = RuleElementMapping
    else:
        raise ValueError(f"Invalid type found in configuration dictionary: {object_type}")
    cls_constructor = mapping[object_type].value
    return cls_constructor(**value_dict)


def construct_ui_rules_from_dict(ui_exposure_settings: Union[dict, DictConfig]) -> UIRules:
    """Takes a dictionary representation of ui exposure logic and constructs an UIRules element out of this.

    Args:
        ui_exposure_settings: dictionary representing the logic to govern exposure of a parameter in the UI

    Returns:
        Exposure element constructed according to the settings passed in ui_exposure_settings
    """
    if ui_exposure_settings is None:
        return NullUIRules()

    rules_dict = ui_exposure_settings.pop("rules", None)
    ui_exposure_settings["rules"] = []
    rules: List[Union[UIRules, Rule]] = []

    if len(rules_dict) == 0:
        return NullUIRules()

    for rule in rules_dict:
        rule_type = str(rule.get(metadata_keys.TYPE, ""))
        if rule_type == RuleElementMapping.UI_RULES.name:
            rules.append(construct_ui_rules_from_dict(rule))
        elif rule_type == RuleElementMapping.RULE.name:
            rules.append(construct_attrib_from_dict(rule))
        else:
            raise ValueError(
                f"Invalid UI exposure settings passed to parser. Configuration contains invalid rule "
                f"type: {rule_type}"
            )
    ui_rules: UIRules = construct_attrib_from_dict(ui_exposure_settings)
    ui_rules.rules = rules
    return ui_rules


def create_default_configurable_enum_from_dict(parameter_dict: Union[dict, DictConfig]) -> dict:
    """Create a default configurable enum from a dictionary.

    Takes a parameter_dict representing a configurable Enum and consumes the ENUM_NAME, OPTIONS and DEFAULT_VALUE
    metadata.

    From this, a new subclass of ConfigurableEnum is constructed. The DEFAULT_VALUE of the parameter_dict is updated
    with the appropriate ConfigurableEnum instance.

    Args:
        parameter_dict: Dictionary representation of an enum_selectable parameter. Can be either serialized or not.

    Returns:
        parameter_dict containing the default_value instantiated as a ConfigurableEnum
    """
    # Some input type validation here first, to keep mypy happy
    if isinstance(parameter_dict, DictConfig):
        param_dict = OmegaConf.to_container(parameter_dict)
        if isinstance(param_dict, dict):
            parameter_dict = param_dict
        else:
            raise TypeError(f"Invalid input parameter_dict of type {type(parameter_dict)}")

    # Create the enum using the functional Enum API. Unfortunately this doesn't play nice with mypy, so ignoring the
    # type error for now
    # pylint: disable=unexpected-keyword-arg
    configurable_enum = ConfigurableEnum(
        parameter_dict.pop(metadata_keys.ENUM_NAME),
        names=parameter_dict.pop(metadata_keys.OPTIONS),
    )  # type: ignore
    # pylint: enable=unexpected-keyword-arg
    serialized_default_value = parameter_dict.pop(metadata_keys.DEFAULT_VALUE)

    if isinstance(serialized_default_value, ConfigurableEnum):
        default_value = serialized_default_value
    else:
        # This instantiates the enum, but mypy doesn't pick up that the created configurable_enum type is callable.
        default_value = configurable_enum(serialized_default_value)  # type: ignore

    parameter_dict.update({metadata_keys.DEFAULT_VALUE: default_value})
    return parameter_dict


def gather_parameter_arguments_and_values_from_dict(config_dict_section: Union[dict, DictConfig]) -> dict:
    """Collect arguments needed to construct attrs class out of a config dict section representing a parameter group.

    Parameters living in the group are constructed in this function as well.

    This function returns a dictionary that contains the keys `make_arguments`, `call_arguments` and `values`.
    make_arguments are the arguments that should be passed to attr.make_class to dynamically generate a class
        constructor
    call_arguments are the arguments that should be passed in the initialization call to the constructor
    values are the parameter values, that can be set once the instance of the parameter group is created.

    Args:
        config_dict_section: Dictionary representation of a parameter
            group in a configuration, for which to gather the
            constructor arguments

    Returns:
        dictionary containing the make_arguments, call_arguments and
        values parsed from the config_dict_section
    """
    # pylint: disable=too-many-locals
    make_arguments = {}
    call_arguments: dict = {}
    all_parameter_values = {}
    for key, value in config_dict_section.items():
        if isinstance(value, (DictConfig, dict)):
            # In case of a nested dict, value represents the settings for a parameter. The arguments are parsed from
            # the dict here and passed to the constructor function for the parameter, according to its `type`.
            parameter_dict = copy.deepcopy(value)
            parameter_type = str(parameter_dict.pop(metadata_keys.TYPE, None))
            if parameter_type == str(ConfigElementType.SELECTABLE):
                parameter_dict = create_default_configurable_enum_from_dict(parameter_dict)
            elif parameter_type == str(None):
                raise ValueError(
                    f"No type was specified for the configurable " f"parameter or parameter group named '{key}'"
                )
            parameter_value = parameter_dict.pop("value", None)

            metadata_enums: Dict[str, Enum] = {}
            for metadata_key, enum_type in METADATA_ENUMS.items():
                enum_value = parameter_dict.pop(metadata_key, None)
                if enum_value is not None:
                    metadata_enums.update({metadata_key: deserialize_enum_value(enum_value, enum_type=enum_type)})

            parameter_ui_rules_dict = parameter_dict.pop(metadata_keys.UI_RULES, None)
            parameter_constructor = PrimitiveElementMapping[parameter_type].value
            parameter_ui_rules = construct_ui_rules_from_dict(parameter_ui_rules_dict)
            parameter_make_arguments = {
                key: parameter_constructor(**parameter_dict, ui_rules=parameter_ui_rules, **metadata_enums)
            }
            make_arguments.update(parameter_make_arguments)
            all_parameter_values.update({key: parameter_value})
        else:
            # Flat values will be passed to the initialization call directly. The only exception is the 'type' item,
            # this should not be included in the call_arguments as it is used to determine the type of constructor for
            # attrs to generate
            if key != metadata_keys.TYPE:
                call_arguments.update({key: value})
    return {
        "make_arguments": make_arguments,
        "call_arguments": call_arguments,
        "values": all_parameter_values,
    }


def create_parameter_group(config_dict_section: Union[dict, DictConfig]) -> ParameterGroupTypeVar:
    """Creates a parameter group object out of a config_dict_section.

    config_dict_section is a dictionary or DictConfig representing a parameter group.
    This method should only be used for simple groups, i.e. parameter groups not containing any other parameter groups.
    For nested groups, the function 'create_nested_parameter_group' should be used instead.

    Args:
        config_dict_section: Dictionary representation of the parameter
            group to construct

    Returns:
        ParameterGroup or ConfigurableParameters object constructed
        according to config_dict_section
    """
    params_and_values = gather_parameter_arguments_and_values_from_dict(config_dict_section)
    make_arguments = params_and_values["make_arguments"]
    call_arguments = params_and_values["call_arguments"]
    all_parameter_values = params_and_values["values"]
    group_type = str(config_dict_section.pop(metadata_keys.TYPE))

    group_constructor_type = GroupElementMapping[group_type].value
    group_constructor = attr.make_class(
        GroupElementMapping[group_type].name,
        bases=(group_constructor_type,),
        attrs=make_arguments,
        eq=False,
        order=False,
    )

    parameter_group = group_constructor(**call_arguments)

    for parameter, value in all_parameter_values.items():
        if value is not None:
            setattr(parameter_group, parameter, value)

    parameter_group.update_auto_hpo_states()
    return parameter_group


def create_nested_parameter_group(config_dict_section: Union[dict, DictConfig]) -> ParameterGroup:
    """Creates a parameter group object out of a config_dict_section.

    config_dict_section is a dictionary or DictConfig representing a parameter group. This method should be used for
    nested groups, and uses recursion to reconstruct those.

    Args:
        config_dict_section: Dictionary representation of the parameter group to construct

    Returns:
        ParameterGroup or Configuration object constructed according to config_dict_section
    """
    groups = {}
    main_config = copy.deepcopy(config_dict_section)

    group_names = contains_parameter_groups(config_dict_section)
    for group_name in group_names:
        group_config_section = config_dict_section.pop(group_name)

        if not contains_parameter_groups(group_config_section):
            childless_parameter_group: ParameterGroup = create_parameter_group(group_config_section)
            groups.update({group_name: childless_parameter_group})

        else:
            parameter_group_with_children: ParameterGroup = create_nested_parameter_group(group_config_section)
            groups.update({group_name: parameter_group_with_children})

        main_config.pop(group_name, None)

    parameter_group: ParameterGroup = create_parameter_group(main_config)

    for group_name, group in groups.items():
        setattr(parameter_group, group_name, group)

    parameter_group.__attrs_post_init__()

    return parameter_group


def contains_parameter_groups(config_dict: Union[dict, DictConfig]) -> List[str]:
    """Checks whether a configuration or configuration section specified in `config_dict` contains parameter groups.

    Returns a list of the group names if it does, and an empty list otherwise

    Args:
        config_dict: Dictionary or DictConfig representing a configuration or configuration section.

    Returns:
        List of names of parameter groups that are defined in the config_dict, if any. Empty list otherwise.
    """
    if isinstance(config_dict, DictConfig):
        input_dict: dict = OmegaConf.to_container(config_dict)  # type: ignore
        # The config_dict will always be converted to a dict by the OmegaConf.to_container call, but this is not
        # reflected by the return type signature. Therefore, mypy can ignore this error
    else:
        input_dict = config_dict

    groups: List[str] = []
    group_category_types: List[str] = [str(x) for x in ConfigElementType if x.category == ElementCategory.GROUPS]

    for field_name, field_value in input_dict.items():
        if isinstance(field_value, dict):
            # If the field is a dict, check whether the type is a parameter group (or derived from it)
            _type = field_value.get(metadata_keys.TYPE, None)
            if str(_type) in group_category_types:
                groups.append(field_name)
    return groups


def from_dict_attr(config_dict: Union[dict, DictConfig]) -> ConfigurableParameters:
    """Creates a configuration object from an input config_dict.

    Uses recursion to handle nested parameter groups in the config

    Args:
        config_dict (Union[dict, DictConfig]): Dictionary representation of a TaskConfig,
            ProjectConfig or ComponentConfig

    Returns:
        ConfigurableParameters: ParameterGroup object constructed according to config_dict
    """
    local_config_dict = copy.deepcopy(config_dict)

    # Initialize all groups in the config, and collect them into the config_groups dict to assign them to the final
    # configuration later on
    config_groups: Dict[str, ParameterGroup] = {}

    # Determine all parameter groups living at the first level of the config
    base_config_groups = contains_parameter_groups(config_dict)
    for group_name in base_config_groups:
        group_config_section = local_config_dict.pop(group_name)
        # If the group itself contains groups, it is nested. If not, it's flat. Initialization for nested and flat
        # groups is different, hence we check this.
        if contains_parameter_groups(group_config_section):
            config_groups.update({group_name: create_nested_parameter_group(group_config_section)})
        else:
            config_groups.update({group_name: create_parameter_group(group_config_section)})

    # Collect parameters for the high level config from the config_dict and create the config constructor, using the
    # type defined in the dict
    config: ConfigurableParameters = create_parameter_group(local_config_dict)

    # Add the groups to the config
    for group_name, group in config_groups.items():
        setattr(config, group_name, group)

    # Run post-initialization to set the groups and parameters attributes correctly
    config.__attrs_post_init__()

    return config


def create(input_config: Union[str, DictConfig, dict]) -> ConfigurableParameters:
    """Create a configuration object from a yaml string, yaml file path, dictionary or OmegaConf DictConfig object.

    Args:
        input_config: yaml string, dictionary, DictConfig or filepath
            describing a configuration.

    Returns:
        ConfigurableParameters object
    """
    # Parse input, validate config type and convert to dict if needed
    config_dict = input_to_config_dict(copy.deepcopy(input_config))
    # Create config from the resulting dictionary
    config: ConfigurableParameters = from_dict_attr(config_dict)

    return config
