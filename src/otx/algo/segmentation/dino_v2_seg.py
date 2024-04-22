# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DinoV2Seg model implementations."""
from __future__ import annotations

from typing import Any

from torch import nn

from otx.algo.segmentation.backbones import DinoVisionTransformer
from otx.algo.segmentation.heads import CustomFCNHead
from otx.core.model.segmentation import OTXSegmentationModel

from .base_model import BaseSegmNNModel


class OTXDinoV2Seg(OTXSegmentationModel):
    """DinoV2Seg Model."""

    def _create_model(self) -> nn.Module:
        backbone = DinoVisionTransformer(**self.backbone_configuration)
        breakpoint()
        decode_head = CustomFCNHead(num_classes=self.num_classes, **self.decode_head_configuration)
        return BaseSegmNNModel(
            backbone=backbone,
            decode_head=decode_head,
            criterion_configuration=self.criterion_configuration,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Seg."""
        return {"model_type": "transformer"}
