"""This module contains utility functions related to Enums."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from enum import Enum
from typing import List, Type


def get_enum_names(enum_cls: Type[Enum]) -> List[str]:
    """Returns a list containing the names of all members of the Enum class passed as `enum_cls`.

    Args:
        enum_cls (Type[Enum]): The Enum class to get the names of its members.

    Returns:
        List[str]: The list of names of all members of the Enum class passed as `enum_cls`.
    """
    return [member.name for member in enum_cls]
