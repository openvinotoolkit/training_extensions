"""This module contains utility functions for use in defining ui rules."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from typing import Union

from otx.api.configuration.ui_rules.types import Action, Operator


def attr_convert_operator(operator: Union[str, Operator]) -> Operator:
    """This function converts an input operator to the correct instance of the Operator Enum.

    It is used when loading an Rule element from a yaml file.
    """
    if isinstance(operator, str):
        return Operator[operator]
    return operator


def attr_convert_action(action: Union[str, Action]) -> Action:
    """This function converts an input action to the correct instance of the Action Enum.

    It is used when loading an Rule element from a yaml file.
    """
    if isinstance(action, str):
        return Action[action]
    return action
