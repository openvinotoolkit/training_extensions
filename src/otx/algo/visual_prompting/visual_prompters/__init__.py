# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Visual prompter modules for OTX visual prompting model."""

from .segment_anything import SegmentAnything, ZeroShotSegmentAnything

__all__ = ["SegmentAnything", "ZeroShotSegmentAnything"]
