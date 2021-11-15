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
This module contains the types used for specifying UI interaction with the configuration
"""

from enum import Enum, auto


class Operator(Enum):
    """
    This Enum represents the allowed operators for use in constructing UI rules for configuration parameters. These
    operators can be used to disable a configuration parameter, conditional on the value of another parameter.
    """

    NOT = auto()
    EQUAL_TO = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    AND = auto()
    OR = auto()
    XOR = auto()

    def __str__(self):
        """
        Retrieves the string representation of an instance of the Enum
        """
        return self.name


class Action(Enum):
    """
    This Enum represents the allowed actions that UI rules can dictate on configuration parameters.
    """

    HIDE = auto()
    SHOW = auto()
    ENABLE_EDITING = auto()
    DISABLE_EDITING = auto()

    def __str__(self):
        """
        Retrieves the string representation of an instance of the Enum
        """
        return self.name
