# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom model implementations for detection task."""

from . import backbones, heads, losses
from .ssd import SSD

__all__ = ["backbones", "heads", "losses", "SSD"]
