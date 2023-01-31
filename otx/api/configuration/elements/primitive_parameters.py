"""This module contains constructor functions for the primitive configurable parameter types.

The available parameter types are: `configurable_integer`, `configurable_float`, `configurable_boolean`,
`string_selectable` and `float_selectable`.
"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from typing import List, Optional, TypeVar, Union

import attr

from otx.api.configuration.enums import AutoHPOState, ConfigElementType, ModelLifecycle
from otx.api.configuration.ui_rules import NullUIRules, UIRules

from .configurable_enum import ConfigurableEnum
from .metadata_keys import (
    AFFECTS_OUTCOME_OF,
    AUTO_HPO_STATE,
    AUTO_HPO_VALUE,
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
    attr_strict_float_converter,
    attr_strict_float_on_setattr,
    attr_strict_int_validator,
    construct_attr_enum_selectable_converter,
    construct_attr_enum_selectable_onsetattr,
    construct_attr_selectable_validator,
    construct_attr_value_validator,
)

# pylint:disable=too-many-arguments

_ConfigurableEnum = TypeVar("_ConfigurableEnum", bound=ConfigurableEnum)


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
    auto_hpo_state: AutoHPOState,
    auto_hpo_value: Optional[Union[int, float, str, bool, ConfigurableEnum]],
) -> dict:
    """Function to construct the dictionary of metadata that is common for all parameter types."""
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
        AUTO_HPO_STATE: auto_hpo_state,
        AUTO_HPO_VALUE: auto_hpo_value,
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
    auto_hpo_state: AutoHPOState = AutoHPOState.NOT_POSSIBLE,
    auto_hpo_value: Optional[int] = None,
) -> int:
    """Constructs a configurable integer attribute, with the appropriate metadata.

    Args:
        default_value: integer to use as default for the parameter
        header: User friendly name for the parameter, which will be
            shown in the UI
        min_value: lower bound of the range of values this parameter can
            take. Defaults to 0
        max_value: upper bound of the range of values this parameter can
            take Defaults to 255
        description: A user friendly description of what this parameter
            does, what does it represent and what are the effects of
            changing it?
        warning: An optional warning message to caution users when
            changing this parameter. This message will be displayed in
            the UI. For example, for the parameter batch_size:
            `Increasing batch size increases GPU memory demands and may
            result in out of memory errors. Please update batch size
            with caution.`
        editable: Set to False to prevent the parameter from being
            edited in the UI. It can still be edited through the REST
            API or the SDK. Defaults to True
        visible_in_ui: Set to False to hide the parameter from the UI
            and the REST API. It will still be visible through the SDK.
            Defaults to True
        affects_outcome_of: Describes the stage of the ModelLifecycle in
            which this parameter modifies the outcome. See the
            documentation for the ModelLifecycle Enum for further
            details
        ui_rules: Set of rules to control UI behavior for this
            parameter. For example, the parameter can be shown or hidden
            from the UI based on the value of other parameters in the
            configuration. Have a look at the UIRules class for more
            details. Defaults to NullUIRules.
        auto_hpo_state: This flag reflects whether the parameter can be
            (or has been) optimized through automatic hyper parameter
            tuning (auto-HPO)
        auto_hpo_value: If auto-HPO has been executed for this
            parameter, this field will hold the optimized value for the
            configurable integer

    Returns:
        attrs Attribute of type `int`, with its metadata set according
        to the inputs
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
        auto_hpo_state=auto_hpo_state,
        auto_hpo_value=auto_hpo_value,
    )

    metadata.update({MIN_VALUE: min_value, MAX_VALUE: max_value})
    value_validator = construct_attr_value_validator(min_value, max_value)
    type_validator = attr_strict_int_validator

    return attr.ib(
        default=default_value,
        type=int,
        validator=[value_validator, type_validator],
        on_setattr=attr.setters.validate,
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
    auto_hpo_state: AutoHPOState = AutoHPOState.NOT_POSSIBLE,
    auto_hpo_value: Optional[float] = None,
) -> float:
    """Constructs a configurable float attribute, with the appropriate metadata.

    Args:
        default_value: float to use as default for the parameter
        header: User friendly name for the parameter, which will be
            shown in the UI
        min_value: lower bound of the range of values this parameter can
            take. Defaults to 0.0
        max_value: upper bound of the range of values this parameter can
            take Defaults to 255.0
        description: A user friendly description of what this parameter
            does, what does it represent and what are the effects of
            changing it?
        warning: An optional warning message to caution users when
            changing this parameter. This message will be displayed in
            the UI. For example, for the parameter batch_size:
            `Increasing batch size increases GPU memory demands and may
            result in out of memory errors. Please update batch size
            with caution.`
        editable: Set to False to prevent the parameter from being
            edited in the UI. It can still be edited through the REST
            API or the SDK. Defaults to True
        visible_in_ui: Set to False to hide the parameter from the UI
            and the REST API. It will still be visible through the SDK.
            Defaults to True
        affects_outcome_of: Describes the stage of the ModelLifecycle in
            which this parameter modifies the outcome. See the
            documentation for the ModelLifecycle Enum for further
            details
        ui_rules: Set of rules to control UI behavior for this
            parameter. For example, the parameter can be shown or hidden
            from the UI based on the value of other parameters in the
            configuration. Have a look at the UIRules class for more
            details. Defaults to NullUIRules.
        auto_hpo_state: This flag reflects whether the parameter can be
            (or has been) optimized through automatic hyper parameter
            tuning (auto-HPO)
        auto_hpo_value: If auto-HPO has been executed for this
            parameter, this field will hold the optimized value for the
            configurable float

    Returns:
        attrs Attribute of type `float`, with its metadata set according
        to the inputs
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
        auto_hpo_state=auto_hpo_state,
        auto_hpo_value=auto_hpo_value,
    )

    metadata.update({MIN_VALUE: min_value, MAX_VALUE: max_value})
    value_validator = construct_attr_value_validator(min_value, max_value)
    type_validator = attr_strict_float_on_setattr

    return attr.ib(
        default=default_value,
        type=float,
        validator=[value_validator, type_validator],
        converter=attr_strict_float_converter,
        on_setattr=[attr.setters.convert, attr.setters.validate],
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
    auto_hpo_state: AutoHPOState = AutoHPOState.NOT_POSSIBLE,
    auto_hpo_value: Optional[bool] = None,
) -> bool:
    """Constructs a configurable boolean attribute, with the appropriate metadata.

    Args:
        default_value: boolean to use as default for the parameter
        header: User friendly name for the parameter, which will be
            shown in the UI
        description: A user friendly description of what this parameter
            does, what does it represent and what are the effects of
            changing it?
        warning: An optional warning message to caution users when
            changing this parameter. This message will be displayed in
            the UI. For example, for the parameter batch_size:
            `Increasing batch size increases GPU memory demands and may
            result in out of memory errors. Please update batch size
            with caution.`
        editable: Set to False to prevent the parameter from being
            edited in the UI. It can still be edited through the REST
            API or the SDK. Defaults to True
        visible_in_ui: Set to False to hide the parameter from the UI
            and the REST API. It will still be visible through the SDK.
            Defaults to True
        affects_outcome_of: Describes the stage of the ModelLifecycle in
            which this parameter modifies the outcome. See the
            documentation for the ModelLifecycle Enum for further
            details
        ui_rules: Set of rules to control UI behavior for this
            parameter. For example, the parameter can be shown or hidden
            from the UI based on the value of other parameters in the
            configuration. Have a look at the UIRules class for more
            details. Defaults to NullUIRules.
        auto_hpo_state: This flag reflects whether the parameter can be
            (or has been) optimized through automatic hyper parameter
            tuning (auto-HPO)
        auto_hpo_value: If auto-HPO has been executed for this
            parameter, this field will hold the optimized value for the
            configurable boolean

    Returns:
        attrs Attribute of type `bool`, with its metadata set according
        to the inputs
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
        auto_hpo_state=auto_hpo_state,
        auto_hpo_value=auto_hpo_value,
    )
    type_validator = attr.validators.instance_of(bool)

    return attr.ib(
        default=default_value,
        metadata=metadata,
        type=bool,
        validator=type_validator,
        on_setattr=attr.setters.validate,
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
    auto_hpo_state: AutoHPOState = AutoHPOState.NOT_POSSIBLE,
    auto_hpo_value: Optional[float] = None,
) -> float:
    """Constructs a configurable float selectable attribute, with the appropriate metadata.

    Args:
        default_value: float to use as default for the parameter
        header: User friendly name for the parameter, which will be
            shown in the UI
        options: list of float options representing the values that this
            parameter can take
        description: A user friendly description of what this parameter
            does, what does it represent and what are the effects of
            changing it?
        warning: An optional warning message to caution users when
            changing this parameter. This message will be displayed in
            the UI. For example, for the parameter batch_size:
            `Increasing batch size increases GPU memory demands and may
            result in out of memory errors. Please update batch size
            with caution.`
        editable: Set to False to prevent the parameter from being
            edited in the UI. It can still be edited through the REST
            API or the SDK. Defaults to True
        visible_in_ui: Set to False to hide the parameter from the UI
            and the REST API. It will still be visible through the SDK.
            Defaults to True
        affects_outcome_of: Describes the stage of the ModelLifecycle in
            which this parameter modifies the outcome. See the
            documentation for the ModelLifecycle Enum for further
            details
        ui_rules: Set of rules to control UI behavior for this
            parameter. For example, the parameter can be shown or hidden
            from the UI based on the value of other parameters in the
            configuration. Have a look at the UIRules class for more
            details. Defaults to NullUIRules.
        auto_hpo_state: This flag reflects whether the parameter can be
            (or has been) optimized through automatic hyper parameter
            tuning (auto-HPO)
        auto_hpo_value: If auto-HPO has been executed for this
            parameter, this field will hold the optimized value for the
            float selectable

    Returns:
        attrs Attribute of type `float`, with its metadata set according
        to the inputs
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
        auto_hpo_state=auto_hpo_state,
        auto_hpo_value=auto_hpo_value,
    )

    metadata.update({OPTIONS: options})
    value_validator = construct_attr_selectable_validator(options)
    type_validator = attr_strict_float_on_setattr

    return attr.ib(
        default=default_value,
        type=float,
        validator=[value_validator, type_validator],
        converter=attr_strict_float_converter,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        metadata=metadata,
    )


def selectable(
    default_value: _ConfigurableEnum,
    header: str,
    description: str = "Default selectable description",
    warning: str = None,
    editable: bool = True,
    visible_in_ui: bool = True,
    affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
    ui_rules: UIRules = NullUIRules(),
    auto_hpo_state: AutoHPOState = AutoHPOState.NOT_POSSIBLE,
    auto_hpo_value: Optional[str] = None,
) -> _ConfigurableEnum:
    """Constructs a selectable attribute from a pre-defined Enum, with the appropriate metadata.

    The list of options for display in the UI is inferred from the type of the ConfigurableEnum instance passed in
    as default_value.

    Args:
        default_value: OTXConfigurationEnum instance to use as default
            for the parameter
        header: User friendly name for the parameter, which will be
            shown in the UI
        description: A user friendly description of what this parameter
            does, what does it represent and what are the effects of
            changing it?
        warning: An optional warning message to caution users when
            changing this parameter. This message will be displayed in
            the UI. For example, for the parameter batch_size:
            `Increasing batch size increases GPU memory demands and may
            result in out of memory errors. Please update batch size
            with caution.`
        editable: Set to False to prevent the parameter from being
            edited in the UI. It can still be edited through the REST
            API or the SDK. Defaults to True
        visible_in_ui: Set to False to hide the parameter from the UI
            and the REST API. It will still be visible through the SDK.
            Defaults to True
        affects_outcome_of: Describes the stage of the ModelLifecycle in
            which this parameter modifies the outcome. See the
            documentation for the ModelLifecycle Enum for further
            details
        ui_rules: Set of rules to control UI behavior for this
            parameter. For example, the parameter can be shown or hidden
            from the UI based on the value of other parameters in the
            configuration. Have a look at the UIRules class for more
            details. Defaults to NullUIRules.
        auto_hpo_state: This flag reflects whether the parameter can be
            (or has been) optimized through automatic hyper parameter
            tuning (auto-HPO)
        auto_hpo_value: If auto-HPO has been executed for this
            parameter, this field will hold the optimized value for the
            string selectable

    Returns:
        attrs Attribute, with its type matching the type of
        `default_value`, and its metadata set according to the inputs
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
        auto_hpo_state=auto_hpo_state,
        auto_hpo_value=auto_hpo_value,
    )

    metadata.update(default_value.get_class_info())

    type_validator = attr.validators.instance_of(ConfigurableEnum)
    value_validator = construct_attr_enum_selectable_onsetattr(default_value)

    # The Attribute returned by attr.ib is not compatible with the return typevar _ConfigurableEnum. However, as the
    # class containing the Attribute is instantiated the selectable type will correspond to the _ConfigurableEnum, so
    # mypy can ignore the error.
    return attr.ib(
        default=default_value,
        type=ConfigurableEnum,
        validator=[type_validator, value_validator],  # type: ignore
        converter=construct_attr_enum_selectable_converter(default_value),
        on_setattr=[attr.setters.convert, value_validator],
        metadata=metadata,
    )  # type: ignore


def string_attribute(value: str) -> str:
    """String attribute.

    Wrapper for attr.ib that can be used to overwrite simple string attributes in a class or parameter group
    definition.

    Args:
        value: string to be added as attribute

    Returns:
        attr.ib string attribute with its default value set to value
    """
    return attr.ib(default=value, type=str, kw_only=True)


def boolean_attribute(value: bool) -> bool:
    """Boolean attribute wrapper.

    Wrapper for attr.ib that can be used to overwrite simple boolean attributes in a class or parameter group
    definition.

    Args:
        value: boolean to be added as attribute

    Returns:
        attr.ib boolean attribute with its default value set to value
    """
    return attr.ib(default=value, type=bool, kw_only=True)
