# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX action classification models."""

from .backbones import MoViNetBackbone
from .heads import MoViNetHead
from .openvino_model import OTXOVActionCls
from .recognizers import BaseRecognizer, MoViNetRecognizer

__all__ = ["OTXOVActionCls", "BaseRecognizer", "MoViNetBackbone", "MoViNetHead", "MoViNetRecognizer"]
