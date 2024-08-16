# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""SegNext model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from otx.algo.segmentation.backbones import MSCAN
from otx.algo.segmentation.heads import LightHamHead
from otx.algo.segmentation.losses import CrossEntropyLossWithIgnore
from otx.algo.segmentation.segmentors import BaseSegmModel
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.segmentation import OTXSegmentationModel

if TYPE_CHECKING:
    from torch import nn


class SegNext(OTXSegmentationModel):
    """SegNext Model."""

    AVAILABLE_MODEL_VERSIONS: ClassVar[list[str]] = [
        "segnext_tiny",
        "segnext_small",
        "segnext_base",
    ]

    def _build_model(self) -> nn.Module:
        # initialize backbones
        if self.model_version not in self.AVAILABLE_MODEL_VERSIONS:
            msg = f"Model version {self.model_version} is not supported."
            raise ValueError(msg)

        backbone = MSCAN(version=self.model_version)
        decode_head = LightHamHead(version=self.model_version, num_classes=self.num_classes)
        criterion = CrossEntropyLossWithIgnore(ignore_index=self.label_info.ignore_index)  # type: ignore[attr-defined]
        return BaseSegmModel(
            backbone=backbone,
            decode_head=decode_head,
            criterion=criterion,
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
