"""Adapters for mmdeploy."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .ops import squeeze__default
from .utils.mmdeploy import is_mmdeploy_enabled

__all__ = [
    "squeeze__default",
    "is_mmdeploy_enabled",
]
