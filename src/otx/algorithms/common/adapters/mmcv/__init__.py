"""Adapters for mmcv support."""

# Copyright (C) 2021-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from .hooks import (
    CancelTrainingHook,
    CheckpointHookWithValResults,
    CustomEvalHook,
    EarlyStoppingHook,
    EMAMomentumUpdateHook,
    EnsureCorrectBestCheckpointHook,
    Fp16SAMOptimizerHook,
    IBLossHook,
    LossDynamicsTrackingHook,
    MemCacheHook,
    NoBiasDecayHook,
    OTXLoggerHook,
    OTXProgressHook,
    ReduceLROnPlateauLrUpdaterHook,
    SAMOptimizerHook,
    SemiSLClsHook,
    StopLossNanTrainingHook,
    TwoCropTransformHook,
)
from .nncf.hooks import CompressionHook
from .nncf.runners import AccuracyAwareRunner
from .ops import multi_scale_deformable_attn_pytorch
from .runner import EpochRunnerWithCancel, IterBasedRunnerWithCancel

__all__ = [
    "EpochRunnerWithCancel",
    "IterBasedRunnerWithCancel",
    "CheckpointHookWithValResults",
    "CustomEvalHook",
    "Fp16SAMOptimizerHook",
    "IBLossHook",
    "SAMOptimizerHook",
    "NoBiasDecayHook",
    "SemiSLClsHook",
    "CancelTrainingHook",
    "OTXLoggerHook",
    "OTXProgressHook",
    "EarlyStoppingHook",
    "ReduceLROnPlateauLrUpdaterHook",
    "EnsureCorrectBestCheckpointHook",
    "StopLossNanTrainingHook",
    "EMAMomentumUpdateHook",
    "CompressionHook",
    "AccuracyAwareRunner",
    "TwoCropTransformHook",
    "MemCacheHook",
    "LossDynamicsTrackingHook",
    "multi_scale_deformable_attn_pytorch",
]
