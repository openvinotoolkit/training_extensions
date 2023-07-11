"""Utility functions for attr package.

This module contains utility functions to use with the attr package, concerning for instance parameter validation
or serialization. They are used within the cfg_helper or the configuration elements.
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from enum import Enum
from typing import Callable, List, Optional, Type, TypeVar, Union

from attr import Attribute

from otx.api.configuration.elements.configurable_enum import ConfigurableEnum
from otx.api.configuration.elements.parameter_group import ParameterGroup
from otx.api.entities.id import ID

NumericTypeVar = TypeVar("NumericTypeVar", int, float)
SelectableTypeVar = TypeVar("SelectableTypeVar", float, str)
ConfigurableEnumTypeVar = TypeVar("ConfigurableEnumTypeVar", bound=ConfigurableEnum)


def attr_enum_to_str_serializer(
    instance: object,  # pylint: disable=unused-argument
    attribute: Attribute,  # pylint: disable=unused-argument
    value: Union[Enum, str],
) -> str:
    """This function converts Enums to their string representation.

    It is used when converting between yaml and python object representation of the configuration.
    The function signature matches what is expected by the `attr.asdict` value_serializer argument.

    Args:
        instance (object): (Unused) The instance of the class that the attribute is attached to.
        attribute (Attribute): (unused) The attribute that is being serialized.
        value (Union[Enum, str]): The value to serialize.

    Returns:
        str: The string representation of the value.
    """
    if isinstance(value, Enum):
        return str(value)
    return value


def _convert_enum_selectable_value(
    value: Union[str, ConfigurableEnumTypeVar],
    enum_class: Type[ConfigurableEnumTypeVar],
) -> ConfigurableEnumTypeVar:
    """Helper function that converts the input value to an instance of the correct ConfigurableEnum.

    Args:
        value (Union[str, ConfigurableEnumTypeVar]): input value to convert
        enum_class (Type[ConfigurableEnumTypeVar]): Type of the Enum to convert to

    Returns:
        ConfigurableEnumTypeVar: Instance of the correct ConfigurableEnum
    """
    if isinstance(value, str):
        try:
            enum_value = enum_class(value)
        except ValueError as ex:
            raise ValueError(
                f"The value {value} is an invalid option for {enum_class.__name__}. Valid options are:"
                f" {enum_class.get_values()}"
            ) from ex
        return enum_value
    return value


def construct_attr_enum_selectable_converter(
    default_value: ConfigurableEnumTypeVar,
) -> Callable[[Union[str, ConfigurableEnumTypeVar]], ConfigurableEnumTypeVar]:
    """This function converts an input value to the correct instance of the ConfigurableEnum.

    It is used when initializing a selectable parameter.

    Args:
        default_value (ConfigurableEnumTypeVar): Default value for the selectable

    Returns:
        Callable[[Union[str, ConfigurableEnumTypeVar]], ConfigurableEnumTypeVar]:
            Function that converts an input value to the correct instance of the ConfigurableEnum.
    """
    enum_class = type(default_value)

    def attr_convert_enum_selectable_value(value: Union[str, ConfigurableEnumTypeVar]) -> ConfigurableEnumTypeVar:
        """Function that converts an input value to an instance of the appropriate ConfigurableEnum.

        Can be used as a `converter` for attrs.Attributes of type ConfigurableEnum.

        Args:
            value (Union[str, ConfigurableEnumTypeVar]): Value to convert to ConfigurableEnum instance

        Returns:
            ConfigurableEnumTypeVar: Instance of the correct ConfigurableEnum
        """
        return _convert_enum_selectable_value(value, enum_class=enum_class)

    return attr_convert_enum_selectable_value


def construct_attr_enum_selectable_onsetattr(
    default_value: ConfigurableEnumTypeVar,
) -> Callable[[ParameterGroup, Attribute, Union[str, ConfigurableEnumTypeVar]], ConfigurableEnumTypeVar]:
    """This function converts an input value to the correct instance of the ConfigurableEnum.

    It is used when setting a value for a selectable parameter.

    Args:
        default_value (ConfigurableEnumTypeVar): Default value for the enum_selectable.

    Returns:
        Callable[[Union[str, ConfigurableEnumTypeVar]], ConfigurableEnumTypeVar]: Function that converts an input
            value to the correct instance of the ConfigurableEnum.
    """
    enum_class = type(default_value)

    def attr_convert_enum_selectable_value(
        instance: ParameterGroup,  # pylint: disable=unused-argument
        attribute: Attribute,  # pylint: disable=unused-argument
        value: Union[str, ConfigurableEnumTypeVar],
    ) -> ConfigurableEnumTypeVar:
        """Function that converts an input value to an instance of the appropriate ConfigurableEnum.

        Can be used with the `on_setattr` hook of the attrs package.
        """
        return _convert_enum_selectable_value(value, enum_class=enum_class)

    return attr_convert_enum_selectable_value


def construct_attr_value_validator(
    min_value: NumericTypeVar, max_value: NumericTypeVar
) -> Callable[[ParameterGroup, Attribute, NumericTypeVar], None]:
    """Constructs a validator function that is used in the attribute validation of numeric configurable parameters.

    Args:
        min_value (NumericTypeVar): Minimum value for the parameter
        max_value (NumericTypeVar): Maximum value for the parameter

    Returns:
        Callable[[ParameterGroup, Attribute, NumericTypeVar], None]: Function that validates the input value.
    """

    def attr_validate_value(
        instance: ParameterGroup, attribute: Attribute, value: NumericTypeVar
    ):  # pylint: disable=unused-argument
        """This function is used to validate values for numeric ConfigurableParameters."""
        if not min_value <= value <= max_value:
            raise ValueError(f"Invalid value set for {attribute.name}: {value} is out of bounds.")

    return attr_validate_value


def construct_attr_selectable_validator(
    options: List[SelectableTypeVar],
) -> Callable[[ParameterGroup, Attribute, SelectableTypeVar], None]:
    """Constructs a validator function that is used in the attribute validation of selectable configurable parameters.

    Args:
        options (List[SelectableTypeVar]): List of valid options for the parameter.

    Returns:
        Callable[[ParameterGroup, Attribute, SelectableTypeVar], None]: Function that validates the input value.
    """

    def attr_validate_selectable(
        instance: ParameterGroup, attribute: Attribute, value: SelectableTypeVar
    ):  # pylint: disable=unused-argument
        """This function is used to validate values for selectable ConfigurableParameters."""
        if value not in options:
            raise ValueError(
                f"Invalid value set for {attribute.name}: {value} is not a valid option for this " f"parameter."
            )

    return attr_validate_selectable


def convert_string_to_id(id_string: Optional[Union[str, ID]]) -> ID:
    """This function converts an input string representing an ID into an OTX ID object.

    Inputs that are already in the form of an ID are left untouched.

    Args:
        id_string (Optional[Union[str, ID]]): string, ID or None object that should be converted to an ID.

    Returns:
        ID: the input as an instance of ID
    """
    if id_string is None:
        output_id = ID()
    elif isinstance(id_string, str):
        output_id = ID(id_string)
    else:
        output_id = id_string
    return output_id


def attr_strict_int_validator(
    instance: ParameterGroup,  # pylint: disable=unused-argument
    attribute: Attribute,
    value: int,
) -> None:
    """Validates that the value set for an attribute is an integer.

    Args:
        instance (ParameterGroup): (Unused) ParameterGroup to which the attribute belongs
        attribute (Attribute): Attribute for which to validate the value
        value (int): Value to validate

    Raises:
        TypeError: if the value passed to the validator is not an integer
    """
    is_strict_int = isinstance(value, int) and not isinstance(value, bool)
    if not is_strict_int:
        raise TypeError(f"Invalid argument type for {attribute.name}: {value} is not of type 'int'")


def _validate_and_convert_float(value: float) -> Optional[float]:
    """Validate that a value is a float, or a number that can be converted to a float.

    If the value is valid, this method will return the value as float. Otherwise, this
    method returns None

    Args:
        value (float): Value to validate and convert

    Returns:
        Optional[float]: The value as a float, or None if the value is not valid
    """
    valid = True
    if not isinstance(value, (float, int)):
        valid = False
    if isinstance(value, bool):
        valid = False
    if valid:
        return float(value)
    return None


def attr_strict_float_on_setattr(
    instance: ParameterGroup,  # pylint: disable=unused-argument
    attribute: Attribute,
    value: float,
) -> float:
    """Validate that the value set for an attribute is a float, or a number that can be converted to a float.

    Args:
        instance (ParameterGroup): ParameterGroup to which the attribute belongs
        attribute (Attribute): Attribute for which to validate the value
        value (float): Value to validate

    Raises:
        TypeError: if the value passed to the validator is not a float or number

    Returns:
        float: The value as a float
    """
    float_value = _validate_and_convert_float(value)
    if float_value is None:
        raise TypeError(f"Invalid argument type for {attribute.name}: {value} is not of type " f"'float'")
    return float_value


def attr_strict_float_converter(value: float) -> float:
    """Converts a value to float.

    Args:
        value (float): value to convert

    Raises:
        TypeError: if value cannot be converted to float

    Returns:
        Value as float
    """
    float_value = _validate_and_convert_float(value)
    if float_value is None:
        raise TypeError(f"Invalid value passed for parameter. Value {value} of type {type(value)} " f"is not a float.")
    return float_value
