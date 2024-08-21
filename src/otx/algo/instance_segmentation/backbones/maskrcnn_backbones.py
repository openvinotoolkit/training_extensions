# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Backbone factory constructor for MaskRCNN model."""
from __future__ import annotations

from functools import partial
from typing import Any, ClassVar

from torch import nn

from otx.algo.common.backbones import ResNet, build_model_including_pytorchcv
from otx.algo.instance_segmentation.backbones import SwinTransformer
from otx.algo.modules.norm import build_norm_layer


class MaskRCNNBackbone:
    """Implementation of MaskRCNN backbone factory for instance segmentation."""

    BACKBONE_CFG: ClassVar[dict[str, Any]] = {
        "maskrcnn_resnet_50": {
            "depth": 50,
            "frozen_stages": 1,
        },
        "maskrcnn_swin_tiny": {
            "drop_path_rate": 0.2,
            "patch_norm": True,
            "convert_weights": True,
        },
        "maskrcnn_efficientnet_b2b": {
            "type": "efficientnet_b2b",
            "out_indices": [2, 3, 4, 5],
            "frozen_stages": -1,
            "pretrained": True,
            "activation": nn.SiLU,
            "normalization": partial(build_norm_layer, nn.BatchNorm2d, requires_grad=True),
        },
    }

    def __new__(cls, model_name: str) -> nn.Module:
        """Create MaskRCNN backbone."""
        if "resnet" in model_name:
            return ResNet(
                **cls.BACKBONE_CFG[model_name],
            )

        if "efficientnet" in model_name:
            return build_model_including_pytorchcv(
                cfg=cls.BACKBONE_CFG[model_name],
            )

        if "swin" in model_name:
            return SwinTransformer(
                **cls.BACKBONE_CFG[model_name],
            )

        msg = ValueError(f"Model {model_name} is not supported.")
        raise msg
