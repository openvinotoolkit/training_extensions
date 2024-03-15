"""OTX Adapters for action recognition backbones - mmaction2."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mmaction.models.backbones.x3d import X3D
from mmdet.models import BACKBONES as MMDET_BACKBONES

from .movinet import OTXMoViNet


def register_action_backbones():
    """Register action backbone to mmdetection backbones."""
    MMDET_BACKBONES.register_module()(X3D)


__all__ = ["OTXMoViNet"]
