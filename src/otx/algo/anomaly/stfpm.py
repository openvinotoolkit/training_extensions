"""OTX STFPM model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

from __future__ import annotations

from typing import Sequence

from anomalib.models import Stfpm as AnomalibStfpm

from otx.core.model.anomaly import AnomalyMixin, OTXAnomaly
from otx.core.types.task import OTXTaskType


class Stfpm(AnomalyMixin, AnomalibStfpm, OTXAnomaly):
    """OTX STFPM model.

    Args:
        layers (Sequence[str]): Feature extractor layers.
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        task (OTXTaskType.ANOMALY_CLASSIFICATION | OTXTaskType.ANOMALY_DETECTION| OTXTaskType.ANOMALY_SEGMENTATION | str
           , optional): Task type of Anomaly Task. CLI passes the task type as a string so it needs to be converted to
            OTXTaskType. Defaults to OTXTaskType.ANOMALY_CLASSIFICATION.
    """

    def __init__(
        self,
        layers: Sequence[str] = ["layer1", "layer2", "layer3"],
        backbone: str = "resnet18",
        task: OTXTaskType.ANOMALY_CLASSIFICATION
        | OTXTaskType.ANOMALY_DETECTION
        | OTXTaskType.ANOMALY_SEGMENTATION
        | str = OTXTaskType.ANOMALY_CLASSIFICATION,
        **kwargs,
    ) -> None:
        super().__init__(layers=layers, backbone=backbone)
        self.task = OTXTaskType(task)
