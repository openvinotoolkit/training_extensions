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
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        layers (list[str], optional): Feature extractor layers. Defaults to ["layer1", "layer2", "layer3"].
        pre_trained (bool, optional): Pretrained backbone. Defaults to True.
        n_features (int | None, optional): Number of features. Defaults to None.
        num_classes (int, optional): Anoamly don't use num_classes ,
            but OTXModel always receives num_classes, so need this.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
        num_classes: int = 2,
    ) -> None:
        OTXAnomaly.__init__(self)
        OTXModel.__init__(self, num_classes=num_classes)
        AnomalibPadim.__init__(
            self,
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            n_features=n_features,
        )
