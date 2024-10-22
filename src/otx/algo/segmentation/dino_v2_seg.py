# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DinoV2Seg model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from otx.algo.segmentation.backbones import DinoVisionTransformer
from otx.algo.segmentation.heads import FCNHead
from otx.algo.segmentation.losses import CrossEntropyLossWithIgnore
from otx.algo.segmentation.segmentors import BaseSegmentationModel
from otx.core.model.segmentation import OTXSegmentationModel

if TYPE_CHECKING:
    from torch import nn
    from typing_extensions import Self


class DinoV2Seg(OTXSegmentationModel):
    """DinoV2Seg Model."""

    AVAILABLE_MODEL_VERSIONS: ClassVar[list[str]] = [
        "dinov2_vits14",
    ]

    def _build_model(self) -> nn.Module:
        if self.model_name not in self.AVAILABLE_MODEL_VERSIONS:
            msg = f"Model version {self.model_name} is not supported."
            raise ValueError(msg)

        backbone = DinoVisionTransformer(model_name=self.model_name, freeze_backbone=True, out_index=[8, 9, 10, 11])
        decode_head = FCNHead(self.model_name, num_classes=self.num_classes)
        criterion = CrossEntropyLossWithIgnore(ignore_index=self.label_info.ignore_index)  # type: ignore[attr-defined]

        return BaseSegmentationModel(
            backbone=backbone,
            decode_head=decode_head,
            criterion=criterion,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Seg."""
        return {"model_type": "transformer"}
