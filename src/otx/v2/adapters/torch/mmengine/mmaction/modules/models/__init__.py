"""OTX Adapters for action recognition models - mmaction2."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .backbones import OTXMoViNet, register_action_backbones
from .heads import MoViNetHead
from .recognizers import MoViNetRecognizer

__all__ = ["register_action_backbones", "OTXMoViNet", "MoViNetHead", "MoViNetRecognizer"]
