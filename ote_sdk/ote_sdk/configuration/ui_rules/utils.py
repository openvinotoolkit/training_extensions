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
This module contains utility functions for use in defining ui rules
"""

from typing import Union

from ote_sdk.configuration.ui_rules.types import Action, Operator


def attr_convert_operator(operator: Union[str, Operator]) -> Operator:
    """
    This function converts an input operator to the correct instance of the Operator Enum. It
    is used when loading an Rule element from a yaml file.
    """
    if isinstance(operator, str):
        return Operator[operator]
    return operator


def attr_convert_action(action: Union[str, Action]) -> Action:
    """
    This function converts an input action to the correct instance of the Action Enum. It
    is used when loading an Rule element from a yaml file.
    """
    if isinstance(action, str):
        return Action[action]
    return action
