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
This module contains the definition for the `validate` function within the configuration helper. This function can be
used to validate the values of a OTE configuration object, checking that each of the parameter values in the
configuration are within their allowed bounds or options.
"""

import attr

from ote_sdk.configuration.elements import ParameterGroup


def _validate_inner(config: ParameterGroup):
    """
    Recursive method that performs validation on all parameters within a parameter group. Uses recursion to validate
    parameters living inside nested groups.

    :raises: ValueError if an out of bounds parameter is found.
    """
    attr.validate(config)
    if config.groups:
        for group_name in config.groups:
            group = getattr(config, group_name)
            _validate_inner(group)


def validate(config: ParameterGroup) -> bool:
    """
    Validate a configuration object

    :param config: Configuration to validate
    :raises: ValueError if the config contains values that are out of bounds
    :return: True if config is valid
    """
    _validate_inner(config)
    return True
