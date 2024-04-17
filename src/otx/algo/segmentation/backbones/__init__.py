# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone modules for OTX segmentation model."""

from .dinov2 import DinoVisionTransformer
from .litehrnet import LiteHRNet
from .mscan import MSCAN

__all__ = ["LiteHRNet", "DinoVisionTransformer", "MSCAN"]
