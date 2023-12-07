# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX classification models."""

from . import backbones
from .otx_dino_v2 import DINOv2RegisterClassifier

__all__ = ["backbones", "DINOv2RegisterClassifier"]
