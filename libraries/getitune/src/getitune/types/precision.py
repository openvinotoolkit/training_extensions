# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getitune precision type definition."""

from enum import Enum


class Precision(str, Enum):
    """getitune precision type definition."""

    FP16 = "FP16"
    FP32 = "FP32"
