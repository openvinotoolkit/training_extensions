"""OTX STFPM model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.model.entity.anomaly import OTXAnomalyModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anomalib.models import AnomalyModule


class STFPM(OTXAnomalyModel):
    """STFPM OTX model."""

    def __init__(
        self,
        layers: Sequence[str],
        input_size: tuple[int, int],
        backbone: str = "resnet18",
        num_classes: int = 2,  # unused as we need only two classes. Kept to match required params.
    ) -> None:
        self.input_size = input_size
        self._layers = layers
        self.backbone = backbone
        super().__init__()

    def _create_model(self) -> AnomalyModule:
        from anomalib.models.image.stfpm.lightning_model import Stfpm

        return Stfpm(layers=self._layers, input_size=self.input_size, backbone=self.backbone)

    @property
    def trainable_model(self) -> str:
        """Used by configure optimizer."""
        return "student_model"
