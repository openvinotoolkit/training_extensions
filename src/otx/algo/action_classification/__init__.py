# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX action classification models."""

from .backbones import OTXMoViNet
from .heads import MoViNetHead
from .openvino_model import OTXOVActionCls
from .recognizers import MoViNetRecognizer, OTXRecognizer3D

__all__ = ["OTXOVActionCls", "OTXRecognizer3D", "OTXMoViNet", "MoViNetHead", "MoViNetRecognizer"]
