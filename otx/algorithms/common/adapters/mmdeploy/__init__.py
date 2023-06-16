"""Adapters for mmdeploy."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from .ops import grid_sampler__default, squeeze__default
from .utils.mmdeploy import is_mmdeploy_enabled

__all__ = [
    "squeeze__default",
    "grid_sampler__default",
    "is_mmdeploy_enabled",
]
