# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base prior generator."""

from __future__ import annotations


class BasePriorGenerator:
    """Base class for prior generator."""

    strides: list[int] | list[tuple[int, int]]
    num_levels: int
