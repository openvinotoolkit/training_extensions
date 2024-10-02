# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""heads modules for 3d object detection."""

from .depth_predictor import DepthPredictor
from .depthaware_transformer import DepthAwareTransformerBuilder

__all__ = ["DepthPredictor", "DepthAwareTransformerBuilder"]
