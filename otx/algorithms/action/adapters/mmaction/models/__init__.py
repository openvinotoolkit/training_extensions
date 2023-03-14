"""OTX Adapters for action recognition models - mmaction2."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .backbones import OTXMoViNet, register_action_backbones
from .detectors import AVAFastRCNN
from .heads import AVARoIHead, MoViNetHead
from .recognizers import MoViNetRecognizer

__all__ = ["register_action_backbones", "AVAFastRCNN", "OTXMoViNet", "MoViNetHead", "MoViNetRecognizer", "AVARoIHead"]
