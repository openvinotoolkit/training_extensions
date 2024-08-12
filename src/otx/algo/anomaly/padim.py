"""OTX Padim model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

from __future__ import annotations

from anomalib.models import Padim as AnomalibPadim

from otx.core.model.anomaly import AnomalyMixin, OTXAnomaly
from otx.core.types.task import OTXTaskType


class Padim(AnomalyMixin, AnomalibPadim, OTXAnomaly):
    """OTX Padim model.

    Args:
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        layers (list[str], optional): Feature extractor layers. Defaults to ["layer1", "layer2", "layer3"].
        pre_trained (bool, optional): Pretrained backbone. Defaults to True.
        n_features (int | None, optional): Number of features. Defaults to None.
        task (OTXTaskType.ANOMALY_CLASSIFICATION | OTXTaskType.ANOMALY_DETECTION| OTXTaskType.ANOMALY_SEGMENTATION | str
           , optional): Task type of Anomaly Task. CLI passes the task type as a string so it needs to be converted to
            OTXTaskType. Defaults to OTXTaskType.ANOMALY_CLASSIFICATION.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
        task: OTXTaskType.ANOMALY_CLASSIFICATION
        | OTXTaskType.ANOMALY_DETECTION
        | OTXTaskType.ANOMALY_SEGMENTATION = OTXTaskType.ANOMALY_CLASSIFICATION,
    ) -> None:
        super().__init__(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            n_features=n_features,
        )
        self.task = OTXTaskType(task)
