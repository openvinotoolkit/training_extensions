"""Adapters for mmdeploy."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from .utils.mmdeploy import is_mmdeploy_enabled

__all__ = [
    "is_mmdeploy_enabled",
]
