# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.


"""
This module contains constructor functions for the primitive configurable parameter types. The available parameter types
are: `configurable_integer`, `configurable_float`, `configurable_boolean`, `string_selectable` and `float_selectable`
"""

from typing import List, Optional, TypeVar, Union

import attr

from ote_sdk.configuration.enums import ConfigElementType, ModelLifecycle
from ote_sdk.configuration.ui_rules import NullUIRules, UIRules

from .configurable_enum import ConfigurableEnum
from .metadata_keys import (
    AFFECTS_OUTCOME_OF,
    DEFAULT_VALUE,
    DESCRIPTION,
    EDITABLE,
    HEADER,
    MAX_VALUE,
    MIN_VALUE,
    OPTIONS,
    TYPE,
    UI_RULES,
    VISIBLE_IN_UI,
    WARNING,
)
from .utils import (
    construct_attr_enum_selectable_converter,
    construct_attr_enum_selectable_onsetattr,
    construct_attr_selectable_validator,
    construct_attr_value_validator,
)

# pylint:disable=too-many-arguments

TConfigurableEnum = TypeVar("TConfigurableEnum", bound=ConfigurableEnum)


def set_common_metadata(
    default_value: Union[int, float, str, bool, ConfigurableEnum],
    header: str,
    description: str,
    warning: Optional[str],
    editable: bool,
    affects_outcome_of: ModelLifecycle,
    ui_rules: UIRules,
    visible_in_ui: bool,
    parameter_type: ConfigElementType,
) -> dict:
    """
    Function to construct the dictionary of metadata that is common for all parameter types
    """
    metadata = {
        DEFAULT_VALUE: default_value,
        DESCRIPTION: description,
        HEADER: header,
        WARNING: warning,
        EDITABLE: editable,
        VISIBLE_IN_UI: visible_in_ui,
        AFFECTS_OUTCOME_OF: affects_outcome_of,
        UI_RULES: ui_rules,
        TYPE: parameter_type,
    }
    return metadata


def configurable_integer(
    default_value: int,
    header: str,
    min_value: int = 0,
    max_value: int = 255,
    description: str = "Default integer description",
    warning: str = None,
    editable: bool = True,
    visible_in_ui: bool = True,
    affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
    ui_rules: UIRules = NullUIRules(),
) -> int:
    """
    Constructs a configurable integer attribute, with the appropriate metadata.

    :param default_value: integer to use as default for the parameter
    :param header: User friendly name for the parameter, which will be shown in the UI
    :param min_value: lower bound of the range of values this parameter can take. Defaults to 0
    :param max_value: upper bound of the range of values this parameter can take Defaults to 255
    :param description: A user friendly description of what this parameter does, what does it represent and what are
        the effects of changing it?
    :param warning: An optional warning message to caution users when changing this parameter. This message will be
        displayed in the UI. For example, for the parameter batch_size: `Increasing batch size increases GPU memory
        demands and may result in out of memory errors. Please update batch size with caution.`
    :param editable: Set to False to prevent the parameter from being edited in the UI. It can still be edited through
        the REST API or the SDK. Defaults to True
    :param visible_in_ui: Set to False to hide the parameter from the UI and the REST API. It will still be visible
        through the SDK. Defaults to True
    :param affects_outcome_of: Describes the stage of the ModelLifecycle in which this parameter modifies the outcome.
        See the documentation for the ModelLifecycle Enum for further details
    :param ui_rules: Set of rules to control UI behavior for this parameter. For example, the parameter can be shown or
        hidden from the UI based on the value of other parameters in the configuration. Have a look at the UIRules
        class for more details. Defaults to NullUIRules.

    :return: attrs Attribute of type `int`, with its metadata set according to the inputs
    """
    metadata = set_common_metadata(
        default_value=default_value,
        header=header,
        description=description,
        warning=warning,
        editable=editable,
        visible_in_ui=visible_in_ui,
        ui_rules=ui_rules,
        affects_outcome_of=affects_outcome_of,
        parameter_type=ConfigElementType.INTEGER,
    )

    metadata.update({MIN_VALUE: min_value, MAX_VALUE: max_value})

    return attr.ib(
        default=default_value,
        type=int,
        validator=[
            attr.validators.instance_of(int),
            construct_attr_value_validator(min_value, max_value),
        ],
        metadata=metadata,
    )


def configurable_float(
    default_value: float,
    header: str,
    min_value: float = 0.0,
    max_value: float = 255.0,
    description: str = "Default float description",
    warning: str = None,
    editable: bool = True,
    visible_in_ui: bool = True,
    affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
    ui_rules: UIRules = NullUIRules(),
) -> float:
    """
    Constructs a configurable float attribute, with the appropriate metadata.

    :param default_value: float to use as default for the parameter
    :param header: User friendly name for the parameter, which will be shown in the UI
    :param min_value: lower bound of the range of values this parameter can take. Defaults to 0.0
    :param max_value: upper bound of the range of values this parameter can take Defaults to 255.0
    :param description: A user friendly description of what this parameter does, what does it represent and what are
        the effects of changing it?
    :param warning: An optional warning message to caution users when changing this parameter. This message will be
        displayed in the UI. For example, for the parameter batch_size: `Increasing batch size increases GPU memory
        demands and may result in out of memory errors. Please update batch size with caution.`
    :param editable: Set to False to prevent the parameter from being edited in the UI. It can still be edited through
        the REST API or the SDK. Defaults to True
    :param visible_in_ui: Set to False to hide the parameter from the UI and the REST API. It will still be visible
        through the SDK. Defaults to True
    :param affects_outcome_of: Describes the stage of the ModelLifecycle in which this parameter modifies the outcome.
        See the documentation for the ModelLifecycle Enum for further details
    :param ui_rules: Set of rules to control UI behavior for this parameter. For example, the parameter can be shown or
        hidden from the UI based on the value of other parameters in the configuration. Have a look at the UIRules
        class for more details. Defaults to NullUIRules.

    :return: attrs Attribute of type `float`, with its metadata set according to the inputs
    """
    metadata = set_common_metadata(
        default_value=default_value,
        header=header,
        description=description,
        warning=warning,
        editable=editable,
        visible_in_ui=visible_in_ui,
        ui_rules=ui_rules,
        affects_outcome_of=affects_outcome_of,
        parameter_type=ConfigElementType.FLOAT,
    )

    metadata.update({MIN_VALUE: min_value, MAX_VALUE: max_value})

    return attr.ib(
        default=default_value,
        type=float,
        validator=construct_attr_value_validator(min_value, max_value),
        converter=float,
        metadata=metadata,
    )


def configurable_boolean(
    default_value: bool,
    header: str,
    description: str = "Default configurable boolean description",
    warning: str = None,
    editable: bool = True,
    visible_in_ui: bool = True,
    affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
    ui_rules: UIRules = NullUIRules(),
) -> bool:
    """
    Constructs a configurable boolean attribute, with the appropriate metadata.

    :param default_value: boolean to use as default for the parameter
    :param header: User friendly name for the parameter, which will be shown in the UI
    :param description: A user friendly description of what this parameter does, what does it represent and what are
        the effects of changing it?
    :param warning: An optional warning message to caution users when changing this parameter. This message will be
        displayed in the UI. For example, for the parameter batch_size: `Increasing batch size increases GPU memory
        demands and may result in out of memory errors. Please update batch size with caution.`
    :param editable: Set to False to prevent the parameter from being edited in the UI. It can still be edited through
        the REST API or the SDK. Defaults to True
    :param visible_in_ui: Set to False to hide the parameter from the UI and the REST API. It will still be visible
        through the SDK. Defaults to True
    :param affects_outcome_of: Describes the stage of the ModelLifecycle in which this parameter modifies the outcome.
        See the documentation for the ModelLifecycle Enum for further details
    :param ui_rules: Set of rules to control UI behavior for this parameter. For example, the parameter can be shown or
        hidden from the UI based on the value of other parameters in the configuration. Have a look at the UIRules
        class for more details. Defaults to NullUIRules.

    :return: attrs Attribute of type `bool`, with its metadata set according to the inputs
    """
    metadata = set_common_metadata(
        default_value=default_value,
        header=header,
        description=description,
        warning=warning,
        editable=editable,
        visible_in_ui=visible_in_ui,
        ui_rules=ui_rules,
        affects_outcome_of=affects_outcome_of,
        parameter_type=ConfigElementType.BOOLEAN,
    )

    return attr.ib(
        default=default_value,
        metadata=metadata,
        type=bool,
        validator=attr.validators.instance_of(bool),
    )


def float_selectable(
    default_value: float,
    header: str,
    options: List[float],
    description: str = "Default selectable description",
    warning: str = None,
    editable: bool = True,
    visible_in_ui: bool = True,
    affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
    ui_rules: UIRules = NullUIRules(),
) -> float:
    """
    Constructs a configurable float selectable attribute, with the appropriate metadata.

    :param default_value: float to use as default for the parameter
    :param header: User friendly name for the parameter, which will be shown in the UI
    :param options: list of float options representing the values that this parameter can take
    :param description: A user friendly description of what this parameter does, what does it represent and what are
        the effects of changing it?
    :param warning: An optional warning message to caution users when changing this parameter. This message will be
        displayed in the UI. For example, for the parameter batch_size: `Increasing batch size increases GPU memory
        demands and may result in out of memory errors. Please update batch size with caution.`
    :param editable: Set to False to prevent the parameter from being edited in the UI. It can still be edited through
        the REST API or the SDK. Defaults to True
    :param visible_in_ui: Set to False to hide the parameter from the UI and the REST API. It will still be visible
        through the SDK. Defaults to True
    :param affects_outcome_of: Describes the stage of the ModelLifecycle in which this parameter modifies the outcome.
        See the documentation for the ModelLifecycle Enum for further details
    :param ui_rules: Set of rules to control UI behavior for this parameter. For example, the parameter can be shown or
        hidden from the UI based on the value of other parameters in the configuration. Have a look at the UIRules
        class for more details. Defaults to NullUIRules.

    :return: attrs Attribute of type `float`, with its metadata set according to the inputs
    """
    metadata = set_common_metadata(
        default_value=default_value,
        header=header,
        description=description,
        warning=warning,
        editable=editable,
        visible_in_ui=visible_in_ui,
        ui_rules=ui_rules,
        affects_outcome_of=affects_outcome_of,
        parameter_type=ConfigElementType.FLOAT_SELECTABLE,
    )

    metadata.update({OPTIONS: options})

    return attr.ib(
        default=default_value,
        type=float,
        validator=construct_attr_selectable_validator(options),
        converter=float,
        metadata=metadata,
    )


def selectable(
    default_value: TConfigurableEnum,
    header: str,
    description: str = "Default selectable description",
    warning: str = None,
    editable: bool = True,
    visible_in_ui: bool = True,
    affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
    ui_rules: UIRules = NullUIRules(),
) -> TConfigurableEnum:
    """
    Constructs a selectable attribute from a pre-defined Enum, with the appropriate metadata. The list of options for
    display in the UI is inferred from the type of the ConfigurableEnum instance passed in as default_value.

    :param default_value: OTEConfigurationEnum instance to use as default for the parameter
    :param header: User friendly name for the parameter, which will be shown in the UI
    :param description: A user friendly description of what this parameter does, what does it represent and what are
        the effects of changing it?
    :param warning: An optional warning message to caution users when changing this parameter. This message will be
        displayed in the UI. For example, for the parameter batch_size: `Increasing batch size increases GPU memory
        demands and may result in out of memory errors. Please update batch size with caution.`
    :param editable: Set to False to prevent the parameter from being edited in the UI. It can still be edited through
        the REST API or the SDK. Defaults to True
    :param visible_in_ui: Set to False to hide the parameter from the UI and the REST API. It will still be visible
        through the SDK. Defaults to True
    :param affects_outcome_of: Describes the stage of the ModelLifecycle in which this parameter modifies the outcome.
        See the documentation for the ModelLifecycle Enum for further details
    :param ui_rules: Set of rules to control UI behavior for this parameter. For example, the parameter can be shown or
        hidden from the UI based on the value of other parameters in the configuration. Have a look at the UIRules
        class for more details. Defaults to NullUIRules.

    :return: attrs Attribute, with its type matching the type of `default_value`, and its metadata set according to
        the inputs
    """
    metadata = set_common_metadata(
        default_value=default_value,
        header=header,
        description=description,
        warning=warning,
        editable=editable,
        visible_in_ui=visible_in_ui,
        ui_rules=ui_rules,
        affects_outcome_of=affects_outcome_of,
        parameter_type=ConfigElementType.SELECTABLE,
    )

    metadata.update(default_value.get_class_info())

    # The Attribute returned by attr.ib is not compatible with the return typevar TConfigurableEnum. However, as the
    # class containing the Attribute is instantiated the selectable type will correspond to the TConfigurableEnum, so
    # mypy can ignore the error.
    return attr.ib(
        default=default_value,
        type=ConfigurableEnum,
        validator=attr.validators.instance_of(ConfigurableEnum),
        converter=construct_attr_enum_selectable_converter(default_value),
        on_setattr=construct_attr_enum_selectable_onsetattr(default_value),
        metadata=metadata,
    )  # type: ignore


def string_attribute(value: str) -> str:
    """
    Wrapper for attr.ib that can be used to overwrite simple string attributes in a class or parameter group definition.

    :param value: string to be added as attribute
    :return: attr.ib string attribute with its default value set to value
    """
    return attr.ib(default=value, type=str, kw_only=True)


def boolean_attribute(value: bool) -> bool:
    """
    Wrapper for attr.ib that can be used to overwrite simple boolean attributes in a class or parameter group
    definition.

    :param value: boolean to be added as attribute
    :return: attr.ib boolean attribute with its default value set to value
    """
    return attr.ib(default=value, type=bool, kw_only=True)
