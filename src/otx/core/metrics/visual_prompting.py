# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX Dice metric used for the OTX semantic segmentation task."""
from __future__ import annotations

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, Dice
from torchmetrics.detection import MeanAveragePrecision

VisualPromptingMetricCallable = lambda label_info: MetricCollection(  # noqa: ARG005
    metrics={
        "iou": BinaryJaccardIndex(),
        "f1-score": BinaryF1Score(),
        "dice": Dice(),
        "mAP": MeanAveragePrecision(iou_type="segm"),
    },
)
