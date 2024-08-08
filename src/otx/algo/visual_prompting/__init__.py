# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX visual prompting models."""

from .segment_anything import SAM
from .zero_shot_segment_anything import OTXZeroShotSegmentAnything, ZeroShotSegmentAnything

__all__ = [
    "SAM",
    "OTXZeroShotSegmentAnything",
    "ZeroShotSegmentAnything",
]
