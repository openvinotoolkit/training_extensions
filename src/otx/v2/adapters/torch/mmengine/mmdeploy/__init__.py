"""Adapters for mmdeploy."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .utils.mmdeploy import is_mmdeploy_enabled

__all__ = [
    "is_mmdeploy_enabled",
]

if is_mmdeploy_enabled():
    from .ops import squeeze__default

    __all__.append("squeeze__default")
