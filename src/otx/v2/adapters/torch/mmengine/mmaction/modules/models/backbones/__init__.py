"""OTX Adapters for action recognition backbones - mmaction2."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mmaction.models.backbones.x3d import X3D
from mmdet.registry import MODELS as MMDET_MODELS

from .movinet import OTXMoViNet


def register_action_backbones() -> None:
    """Register action backbone to mmdetection backbones."""
    MMDET_MODELS.register_module()(X3D)


__all__ = ["OTXMoViNet"]
