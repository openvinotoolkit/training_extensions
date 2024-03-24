# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Plugin for mixed-precision training on XPU."""

from .xpu_precision import MixedPrecisionXPUPlugin

__all__ = ["MixedPrecisionXPUPlugin"]
