"""OTX Padim model."""


# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from anomalib.models.image import Padim as AnomalibPadim

from otx.core.model.entity.base import OTXModel
from otx.core.model.module.anomaly import OTXAnomaly


class Padim(OTXAnomaly, OTXModel, AnomalibPadim):
    """OTX Padim model.

    Args:
        input_size (tuple[int, int], optional): Input size. Defaults to (256, 256).
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        layers (list[str], optional): Feature extractor layers. Defaults to ["layer1", "layer2", "layer3"].
        pre_trained (bool, optional): Pretrained backbone. Defaults to True.
        n_features (int | None, optional): Number of features. Defaults to None.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (256, 256),
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
    ) -> None:
        OTXAnomaly.__init__(self)
        OTXModel.__init__(self, num_classes=2)
        AnomalibPadim.__init__(
            self,
            input_size=input_size,
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            n_features=n_features,
        )
