# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Plugins for the Ultralytics backend."""

from .xpu_mixin import XPUAwareTrainerMixin

__all__ = ["XPUAwareTrainerMixin"]
