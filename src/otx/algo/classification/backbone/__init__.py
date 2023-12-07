# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone modules for OTX custom model."""

from .otx_efficientnet import OTXEfficientNet
from .otx_efficientnet_v2 import OTXEfficientNetV2
from .otx_mobilenet_v3 import OTXMobileNetV3

__all__ = ["OTXEfficientNet", "OTXEfficientNetV2", "OTXMobileNetV3"]
