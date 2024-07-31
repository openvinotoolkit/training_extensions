# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Bbox structures for instance segmentation task."""

from .transforms import (
    bbox2roi,
    empty_box_as,
    get_box_wh,
    scale_boxes,
)

__all__ = [
    "bbox2roi",
    "empty_box_as",
    "get_box_wh",
    "scale_boxes",
]
