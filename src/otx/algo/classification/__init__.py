# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX classification models."""

from . import backbones, heads, losses
from .otx_dino_v2 import DINOv2, DINOv2RegisterClassifier

__all__ = ["backbones", "heads", "losses", "DINOv2", "DINOv2RegisterClassifier"]
