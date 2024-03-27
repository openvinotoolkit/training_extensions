"""Definitions for `substitute_values` and `substitute_values_for_lifecycle` functions within the configuration helper.

These functions can be used to update values or ids in a OTX configuration object, according to a value dictionary or
configuration object
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from typing import Dict, Optional, Sequence, Union

from omegaconf import DictConfig

from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.elements import ParameterGroup, metadata_keys
from otx.api.configuration.enums import ModelLifecycle

from .convert import convert
from .create import input_to_config_dict
from .utils import search_in_config_dict
from .validate import validate


def _should_parameter_be_updated(
    config_element: ParameterGroup,
    parameter_name: str,
    model_lifecycle: Optional[Union[Sequence[ModelLifecycle], ModelLifecycle]],
) -> bool:
    """Check if parameter should be updated.

    Checks whether a parameter `parameter_name` belonging to a ParameterGroup
    `config_element` fullfills the criterion for updating it's value, giving in
    `model_lifecycle`.

    config_element (ParameterGroup): ParameterGroup which the parameter belongs to
    parameter_name (str): Name of the parameter
    model_lifecycle (Optional[Union[Sequence[ModelLifecycle], ModelLifecycle]]):
        Phase or list of phases of the model lifecycle which the parameter should
        affect in order to update it.

    Returns:
        bool: True if the parameter should be updated, False otherwise
    """
    should_update = True
    if model_lifecycle is not None:
        # Skip parameter substitution if model lifecycle does not match
        # provided value
        metadata = config_element.get_metadata(parameter_name)
        parameter_affects_outcome_of = metadata[metadata_keys.AFFECTS_OUTCOME_OF]
        if isinstance(model_lifecycle, list):
            should_update = parameter_affects_outcome_of in model_lifecycle
        else:
            should_update = parameter_affects_outcome_of == model_lifecycle
    return should_update


def _substitute(
    config: ConfigurableParameters,
    value_input: Dict,
    allow_missing_values: bool = False,
    model_lifecycle: Optional[Union[ModelLifecycle, Sequence[ModelLifecycle]]] = None,
):
    """Substitutes values from value_input into the config object.

    The structures of value_input and config have to match in order for the values to be substituted
    correctly. If the argument `model_lifecycle` is provided, only parameters that
    affect the specified phase in the model lifecycle will be substituted.

    Values are substituted in place.

    Args:
        config (ConfigurableParameters): ConfigurableParameter object to substitute values into.
        value_input (Dict): ConfigurableParameters (either in object, dict, yaml or DictConfig representation) to take
            the values to be substituted from.
        allow_missing_values (bool): True to allow missing values in the configuration, i.e. if a value is found in
            `value_input`, but not in `config`, it will silently be ignored. If set to False, an AttributeError will be
            raised. Defaults to False.
        model_lifecycle (Optional[Union[ModelLifecycle, Sequence[ModelLifecycle]]]): Optional phase or list of phases
            in the model lifecycle to carry out the substitution for. If no `model_lifecycle` is provided, substitution
            will be carried out for all parameters.
    """
    # Search all 'value' entries in the input dict
    values_and_paths_list = search_in_config_dict(value_input, key_to_search="value")

    # Substitute the values in the config, according to their paths
    for (value, path) in values_and_paths_list:
        if metadata_keys.UI_RULES in path:
            # Skip entries that involve ui rules (these entries also have a `value` key), since we are only
            # concerned about parameter values here
            continue
        config_element = config

        for attribute_name in path:
            if attribute_name != path[-1]:
                # Traverse the config according to path
                if hasattr(config_element, attribute_name):
                    config_element = getattr(config_element, attribute_name)
                else:
                    if not allow_missing_values:
                        raise AttributeError(
                            f"Configuration does not match structure of the values to "
                            f"substitute from. The configuration {config_element} does "
                            f"not contain the parameter group {attribute_name}"
                        )
            else:
                # At the end of the path, update the attribute value
                if hasattr(config_element, attribute_name):
                    if _should_parameter_be_updated(
                        config_element=config_element,
                        parameter_name=attribute_name,
                        model_lifecycle=model_lifecycle,
                    ):
                        setattr(config_element, attribute_name, value)
                else:
                    if not allow_missing_values:
                        raise AttributeError(
                            f"Configuration does not match structure of the values "
                            f"to substitute from. Target parameter group "
                            f"{config_element} does not contain the parameter "
                            f"{attribute_name}."
                        )


def substitute_values(
    config: ConfigurableParameters,
    value_input: Union[str, DictConfig, dict, ConfigurableParameters],
    allow_missing_values: bool = False,
):
    """Substitutes values from value_input into the config object.

    The structures of value_input and config have to match in order for the values to be substituted
    correctly.

    Values are substituted in place.

    Args:
        config (ConfigurableParameters): ConfigurableParameter object to substitute values into.
        value_input (Union[str, DictConfig, dict, ConfigurableParameters]): ConfigurableParameters (either in object,
            dict, yaml or DictConfig representation) to take the values to be substituted from.
        allow_missing_values (bool): True to allow missing values in the configuration,
            i.e. if a value is found in `value_input`, but not in `config`, it will
            silently be ignored. If set to False, an AttributeError will be
            raised. Defaults to False.
    """
    # Parse input and convert to dict if needed
    if isinstance(value_input, ConfigurableParameters):
        input_dict: Dict = convert(value_input, dict)
    else:
        input_dict = input_to_config_dict(value_input, check_config_type=False)

    _substitute(config, input_dict, allow_missing_values=allow_missing_values)

    # Finally, validate the config with the updated values
    validate(config)


def substitute_values_for_lifecycle(
    config: ConfigurableParameters,
    value_input: ConfigurableParameters,
    model_lifecycle: Union[ModelLifecycle, Sequence[ModelLifecycle]],
    allow_missing_values: bool = True,
):
    """Substitutes values from value_input into the config object.

    The structures of value_input and config have to match in order for the values to be substituted correctly.
    If the argument `model_lifecycle` is provided, only parameters that affect the specified phase in the model
    lifecycle will be substituted.

    Values are substituted in place.

    Args:
        config (ConfigurableParameters): ConfigurableParameter object to substitute values into
        value_input (ConfigurableParameters): ConfigurableParameters to take the values to be substituted from.
        model_lifecycle (Union[ModelLifecycle, Sequence[ModelLifecycle]]): Phase or list of phases in the
            model lifecycle to carry out the substitution for. For example, if
            `model_lifecycle = ModelLifecycle.INFERENCE` is passed, only parameters that
            affect inference will be updated, and the rest of the parameters will
            remain untouched.
        allow_missing_values (bool): True to allow missing values in the configuration,
            i.e. if a value is found in `value_input`, but not in `config`, it will
            silently be ignored. If set to False, an AttributeError will be
            raised. Defaults to True.
    """
    input_dict: dict = convert(value_input, dict)

    _substitute(
        config,
        input_dict,
        allow_missing_values=allow_missing_values,
        model_lifecycle=model_lifecycle,
    )

    # Finally, validate the config with the updated values
    validate(config)
