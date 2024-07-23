# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DinoV2Seg model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from otx.algo.segmentation.backbones import DinoVisionTransformer
from otx.algo.segmentation.heads import FCNHead
from otx.core.model.segmentation import TorchVisionCompatibleModel

from .base_model import BaseSegmModel

if TYPE_CHECKING:
    from torch import nn
    from typing_extensions import Self


class DinoV2Seg(BaseSegmModel):
    """DinoV2Seg Model."""

    default_backbone_configuration: ClassVar[dict[str, Any]] = {
        "name": "dinov2_vits14",
        "freeze_backbone": True,
        "out_index": [8, 9, 10, 11],
    }
    default_decode_head_configuration: ClassVar[dict[str, Any]] = {
        "norm_cfg": {"type": "SyncBN", "requires_grad": True},
        "in_channels": [384, 384, 384, 384],
        "in_index": [0, 1, 2, 3],
        "input_transform": "resize_concat",
        "channels": 1536,
        "kernel_size": 1,
        "num_convs": 1,
        "concat_input": False,
        "dropout_ratio": -1,
        "align_corners": False,
        "pretrained_weights": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_ade20k_linear_head.pth",
    }


class OTXDinoV2Seg(TorchVisionCompatibleModel):
    """DinoV2Seg Model."""

    def _create_model(self) -> nn.Module:
        # merge configurations with defaults overriding them
        backbone_configuration = DinoV2Seg.default_backbone_configuration | self.backbone_configuration
        decode_head_configuration = DinoV2Seg.default_decode_head_configuration | self.decode_head_configuration
        backbone = DinoVisionTransformer(**backbone_configuration)
        decode_head = FCNHead(num_classes=self.num_classes, **decode_head_configuration)
        return DinoV2Seg(
            backbone=backbone,
            decode_head=decode_head,
            criterion_configuration=self.criterion_configuration,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Seg."""
        return {"model_type": "transformer"}

    def to(self, *args, **kwargs) -> Self:
        """Return a model with specified device."""
        ret = super().to(*args, **kwargs)
        if self.device.type == "xpu":
            msg = f"{type(self).__name__} doesn't support XPU."
            raise RuntimeError(msg)
        return ret
