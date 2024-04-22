# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""SegNext model implementations."""
from __future__ import annotations

from typing import Any

from torch import nn

from otx.algo.segmentation.backbones import MSCAN

from otx.algo.segmentation.heads import LightHamHead
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.segmentation import OTXSegmentationModel

from .base_model import BaseSegmNNModel


class OTXSegNext(OTXSegmentationModel):
    """SegNext Model."""

    def _create_model(self) -> nn.Module:
        backbone = MSCAN(**self.backbone_configuration)
        breakpoint()
        decode_head = LightHamHead(num_classes=self.num_classes, **self.decode_head_configuration)
        return BaseSegmNNModel(
            backbone=backbone,
            decode_head=decode_head,
            pretrained_weights=self.pretrained_weights,
            criterion_configuration=self.criterion_configuration,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_seg_segnext_ckpt(state_dict, add_prefix)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for SegNext."""
        # TODO(Kirill): check PTQ removing hamburger from ignored_scope
        return {
            "ignored_scope": {
                "patterns": ["__module.model.decode_head.hamburger*"],
                "types": [
                    "Add",
                    "MVN",
                    "Divide",
                    "Multiply",
                ],
            },
        }
