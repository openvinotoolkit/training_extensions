# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX Dice metric used for the OTX semantic segmentation task."""
from torchmetrics.classification.dice import Dice
from torchmetrics.collections import MetricCollection

from otx.core.types.label import LabelInfo


def _dice_callable(label_info: LabelInfo) -> MetricCollection:
    return MetricCollection(
        {"Dice": Dice(num_classes=label_info.num_classes + 1, ignore_index=label_info.num_classes)},
    )


DiceCallable = _dice_callable
