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
This module contains all elements needed to construct a OTE configuration object
"""


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
