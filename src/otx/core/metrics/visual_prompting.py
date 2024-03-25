# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX Dice metric used for the OTX semantic segmentation task."""
from __future__ import annotations

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, Dice
from torchmetrics.detection import MeanAveragePrecision

from otx.core.types.label import LabelInfo


def _visual_prompting_metric_callable(label_info: LabelInfo) -> MetricCollection:  # noqa: ARG001
    return MetricCollection(
        metrics={
            "iou": BinaryJaccardIndex(),
            "f1-score": BinaryF1Score(),
            "dice": Dice(),
            "mAP": MeanAveragePrecision(iou_type="segm"),
        },
    )


VisualPromptingMetricCallable = _visual_prompting_metric_callable
