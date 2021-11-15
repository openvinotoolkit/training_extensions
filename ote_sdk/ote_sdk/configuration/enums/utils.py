"""
This module contains utility functions related to Enums
"""

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

from enum import Enum
from typing import List, Type


def get_enum_names(enum_cls: Type[Enum]) -> List[str]:
    """
    Returns a list containing the names of all members of the Enum class passed as
    `enum_cls`.

    :param enum_cls: Type of the enum to retrieve the names for
    :return: list of Enum members names
    """
    return [member.name for member in enum_cls]
