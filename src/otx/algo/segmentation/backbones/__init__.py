# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone modules for OTX segmentation model."""

from .litehrnet import LiteHRNet
from .dinov2 import DinoVisionTransformer

__all__ = ["LiteHRNet", "DinoVisionTransformer"]
