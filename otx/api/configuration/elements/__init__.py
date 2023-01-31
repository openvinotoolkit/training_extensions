"""This module contains all elements needed to construct a OTX configuration object."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from .configurable_enum import ConfigurableEnum
from .parameter_group import ParameterGroup, add_parameter_group
from .primitive_parameters import (
    boolean_attribute,
    configurable_boolean,
    configurable_float,
    configurable_integer,
    float_selectable,
    selectable,
    string_attribute,
)

__all__ = [
    "ConfigurableEnum",
    "ParameterGroup",
    "add_parameter_group",
    "boolean_attribute",
    "configurable_boolean",
    "configurable_float",
    "configurable_integer",
    "float_selectable",
    "selectable",
    "string_attribute",
]
