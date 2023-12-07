# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom head implementations for detection task."""

from .custom_anchor_generator import SSDAnchorGeneratorClustered
from .custom_ssd_head import CustomSSDHead

__all__ = ["SSDAnchorGeneratorClustered", "CustomSSDHead"]
