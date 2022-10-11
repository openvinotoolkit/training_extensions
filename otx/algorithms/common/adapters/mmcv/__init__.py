"""Adapters for mmcv support"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .hooks import (
    CancelTrainingHook,
    EarlyStoppingHook,
    EnsureCorrectBestCheckpointHook,
    OTELoggerHook,
    OTEProgressHook,
    ReduceLROnPlateauLrUpdaterHook,
    StopLossNanTrainingHook,
)
from .runner import EpochRunnerWithCancel, IterBasedRunnerWithCancel

__all__ = [
    "EpochRunnerWithCancel",
    "IterBasedRunnerWithCancel",
    "CancelTrainingHook",
    "OTELoggerHook",
    "OTEProgressHook",
    "EarlyStoppingHook",
    "ReduceLROnPlateauLrUpdaterHook",
    "EnsureCorrectBestCheckpointHook",
    "StopLossNanTrainingHook",
]
