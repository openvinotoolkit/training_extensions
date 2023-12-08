"""Plugin for mixed-precision training on XPU."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .xpu_precision import MixedPrecisionXPUPlugin

__all__ = ["MixedPrecisionXPUPlugin"]
