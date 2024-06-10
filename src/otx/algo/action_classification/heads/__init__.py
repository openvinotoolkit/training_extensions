# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom heads for action classification."""

from .movinet_head import MoViNetHead
from .x3d_head import X3DHead

__all__ = ["MoViNetHead", "X3DHead"]
