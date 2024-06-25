# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Coders for detection task."""

from .delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from .distance_point_bbox_coder import DistancePointBBoxCoder

__all__ = ["DeltaXYWHBBoxCoder", "DistancePointBBoxCoder"]
