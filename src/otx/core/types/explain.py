# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX explain type definition."""

from __future__ import annotations

from enum import Enum


class TargetExplainGroup(str, Enum):
    """OTX target explain group definition."""

    IMAGE = "IMAGE"
    ALL = "ALL"
    PREDICTIONS = "PREDICTIONS"
    CUSTOM = "CUSTOM"
