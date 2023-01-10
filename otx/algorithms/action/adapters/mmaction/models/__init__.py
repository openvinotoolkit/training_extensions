"""OTX Adapters for action recognition models - mmaction2."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .backbones import register_action_backbones
from .detectors import AVAFastRCNN
from .heads import AVARoIHead

__all__ = ["register_action_backbones", "AVAFastRCNN", "AVARoIHead"]
