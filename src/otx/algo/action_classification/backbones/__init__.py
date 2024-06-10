# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom backbones for action classification."""

from .movinet import MoViNetBackbone
from .x3d import X3DBackbone

__all__ = ["MoViNetBackbone", "X3DBackbone"]
