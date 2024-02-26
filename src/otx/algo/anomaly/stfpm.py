"""OTX STFPM model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from anomalib.models.image.stfpm import Stfpm as AnomalibStfpm

from otx.core.model.entity.base import OTXModel
from otx.core.model.module.anomaly import OTXAnomaly

if TYPE_CHECKING:
    from collections.abc import Sequence


class Stfpm(OTXAnomaly, OTXModel, AnomalibStfpm):
    """OTX STFPM model.

    Args:
        layers (Sequence[str]): Feature extractor layers.
        input_size (tuple[int, int]): Input size.
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
    """

    def __init__(
        self,
        layers: Sequence[str] = ["layer1", "layer2", "layer3"],
        input_size: tuple[int, int] = (256, 256),
        backbone: str = "resnet18",
    ) -> None:
        OTXAnomaly.__init__(self)
        OTXModel.__init__(self, num_classes=2)
        AnomalibStfpm.__init__(
            self,
            input_size=input_size,
            backbone=backbone,
            layers=layers,
        )

    @property
    def trainable_model(self) -> str:
        """Used by configure optimizer."""
        return "student_model"
