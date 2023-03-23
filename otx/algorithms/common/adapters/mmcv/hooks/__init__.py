"""Adapters for mmcv support."""

# Copyright (C) 2021-2023 Intel Corporation
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

from .adaptive_training_hooks import AdaptiveTrainSchedulingHook
from .cancel_hook import CancelInterfaceHook, CancelTrainingHook
from .checkpoint_hook import (
    CheckpointHookWithValResults,
    EnsureCorrectBestCheckpointHook,
)
from .composed_dataloaders_hook import ComposedDataLoadersHook
from .early_stopping_hook import (
    EarlyStoppingHook,
    LazyEarlyStoppingHook,
    ReduceLROnPlateauLrUpdaterHook,
    StopLossNanTrainingHook,
)
from .ema_momentum_update_hook import EMAMomentumUpdateHook
from .eval_hook import CustomEvalHook, DistCustomEvalHook
from .force_train_hook import ForceTrainModeHook
from .fp16_sam_optimizer_hook import Fp16SAMOptimizerHook
from .ib_loss_hook import IBLossHook
from .logger_hook import OTXLoggerHook
from .logger_replace_hook import LoggerReplaceHook
from .model_ema_hook import CustomModelEMAHook, DualModelEMAHook
from .model_ema_v2_hook import ModelEmaV2Hook
from .no_bias_decay_hook import NoBiasDecayHook
from .progress_hook import OTXProgressHook
from .recording_forward_hooks import (
    ActivationMapHook,
    BaseRecordingForwardHook,
    EigenCamHook,
    FeatureVectorHook,
)
from .sam_optimizer_hook import SAMOptimizerHook
from .save_initial_weight_hook import SaveInitialWeightHook
from .semisl_cls_hook import SemiSLClsHook
from .task_adapt_hook import TaskAdaptHook
from .two_crop_transform_hook import TwoCropTransformHook
from .unbiased_teacher_hook import UnbiasedTeacherHook
from .workflow_hooks import WorkflowHook

__all__ = [
    "AdaptiveTrainSchedulingHook",
    "CancelInterfaceHook",
    "CancelTrainingHook",
    "CheckpointHookWithValResults",
    "EnsureCorrectBestCheckpointHook",
    "ComposedDataLoadersHook",
    "CustomEvalHook",
    "DistCustomEvalHook",
    "EarlyStoppingHook",
    "LazyEarlyStoppingHook",
    "ReduceLROnPlateauLrUpdaterHook",
    "EMAMomentumUpdateHook",
    "ForceTrainModeHook",
    "Fp16SAMOptimizerHook",
    "StopLossNanTrainingHook",
    "IBLossHook",
    "OTXLoggerHook",
    "LoggerReplaceHook",
    "CustomModelEMAHook",
    "DualModelEMAHook",
    "ModelEmaV2Hook",
    "NoBiasDecayHook",
    "OTXProgressHook",
    "BaseRecordingForwardHook",
    "EigenCamHook",
    "ActivationMapHook",
    "FeatureVectorHook",
    "SAMOptimizerHook",
    "SaveInitialWeightHook",
    "SemiSLClsHook",
    "TaskAdaptHook",
    "TwoCropTransformHook",
    "UnbiasedTeacherHook",
    "WorkflowHook",
]
