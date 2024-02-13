"""OTX Padim model."""


# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.model.entity.anomaly import OTXAnomalyModel

if TYPE_CHECKING:
    from torch import nn


class Padim(OTXAnomalyModel):
    """Padim OTX model."""

    def __init__(
        self,
        input_size: tuple[int, int],
        layers: list[str],
        backbone: str = "resnet18",
        pre_trained: bool = True,
        n_features: int | None = None,
        num_classes: int = 2,  # unused as we need only two classes. Kept to match required params.
    ) -> None:
        self.input_size = input_size
        self._layers = layers
        self.backbone = backbone
        self.pre_trained = pre_trained
        self.n_features = n_features
        super().__init__()

        self.anomalib_lightning_args.update(
            input_size=self.input_size,
            layers=self._layers,
            backbone=self.backbone,
            pre_trained=self.pre_trained,
            n_features=self.n_features,
        )

    def _create_model(self) -> nn.Module:
        from anomalib.models.image.padim.torch_model import PadimModel

        return PadimModel(
            input_size=self.input_size,
            layers=self._layers,
            backbone=self.backbone,
            pre_trained=self.pre_trained,
            n_features=self.n_features,
        )
