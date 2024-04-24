# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""SegNext model implementations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.algo.segmentation.backbones import MSCAN
from otx.algo.segmentation.heads import LightHamHead
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.segmentation import TorchVisionCompatibleModel

from .base_model import BaseSegmNNModel

if TYPE_CHECKING:
    from torch import nn


class SegNextB(BaseSegmNNModel):
    """SegNextB Model."""


class SegNextS(BaseSegmNNModel):
    """SegNextS Model."""


class SegNextT(BaseSegmNNModel):
    """SegNextT Model."""


SEGNEXT_VARIANTS = {
    "SegNextB": SegNextB,
    "SegNextS": SegNextS,
    "SegNextT": SegNextT,
}


class OTXSegNext(TorchVisionCompatibleModel):
    """SegNext Model."""

    def _create_model(self) -> nn.Module:
        backbone = MSCAN(**self.backbone_configuration)
        decode_head = LightHamHead(num_classes=self.num_classes, **self.decode_head_configuration)
        return SEGNEXT_VARIANTS[self.name_base_model](
            backbone=backbone,
            decode_head=decode_head,
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
