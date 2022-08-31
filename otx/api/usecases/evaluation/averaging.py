"""Averaging module contains averaging method enumeration."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from enum import Enum, auto


class MetricAverageMethod(Enum):
    """This defines the metrics averaging method."""

    MICRO = auto()
    MACRO = auto()
