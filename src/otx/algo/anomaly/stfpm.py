"""OTX STFPM model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.model.entity.anomaly import OTXAnomalyModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import nn


class STFPM(OTXAnomalyModel):
    """STFPM OTX model."""

    def __init__(
        self,
        layers: Sequence[str],
        input_size: tuple[int, int],
        backbone: str = "resnet18",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.layers = layers
        self.backbone = backbone

        self.anomalib_lightning_args.update(
            input_size=self.input_size,
            layers=self.layers,
            backbone=self.backbone,
        )

    def _create_model(self) -> nn.Module:
        from anomalib.models.image.stfpm.torch_model import STFPMModel

        return STFPMModel(layers=self.layers, input_size=self.input_size, backbone=self.backbone)
