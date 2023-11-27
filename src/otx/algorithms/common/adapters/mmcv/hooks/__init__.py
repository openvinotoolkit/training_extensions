"""Adapters for mmcv support."""

# Copyright (C) 2022-2023 Intel Corporation
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

from .adaptive_repeat_data_hook import AdaptiveRepeatDataHook
from .adaptive_training_hook import AdaptiveTrainSchedulingHook
from .cancel_hook import CancelInterfaceHook, CancelTrainingHook
from .checkpoint_hook import (
    CheckpointHookWithValResults,
    EnsureCorrectBestCheckpointHook,
    SaveInitialWeightHook,
)
from .composed_dataloaders_hook import ComposedDataLoadersHook
from .custom_model_ema_hook import CustomModelEMAHook, EMAMomentumUpdateHook
from .dual_model_ema_hook import DualModelEMAHook
from .early_stopping_hook import (
    EarlyStoppingHook,
    LazyEarlyStoppingHook,
    ReduceLROnPlateauLrUpdaterHook,
    StopLossNanTrainingHook,
)
from .eval_hook import CustomEvalHook, DistCustomEvalHook
from .force_train_hook import ForceTrainModeHook
from .fp16_sam_optimizer_hook import Fp16SAMOptimizerHook
from .ib_loss_hook import IBLossHook
from .logger_hook import LoggerReplaceHook, OTXLoggerHook
from .loss_dynamics_tracking_hook import LossDynamicsTrackingHook
from .mean_teacher_hook import MeanTeacherHook
from .mem_cache_hook import MemCacheHook
from .model_ema_v2_hook import ModelEmaV2Hook
from .no_bias_decay_hook import NoBiasDecayHook
from .progress_hook import OTXProgressHook
from .recording_forward_hook import (
    ActivationMapHook,
    BaseRecordingForwardHook,
    EigenCamHook,
    FeatureVectorHook,
)
from .sam_optimizer_hook import SAMOptimizerHook
from .semisl_cls_hook import SemiSLClsHook
from .task_adapt_hook import TaskAdaptHook
from .two_crop_transform_hook import TwoCropTransformHook

__all__ = [
    "AdaptiveRepeatDataHook",
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
    "MeanTeacherHook",
    "MemCacheHook",
    "LossDynamicsTrackingHook",
]

try:
    from .hpu_optimizer_hook import HPUOptimizerHook

    __all__ += ["HPUOptimizerHook"]
except:  # noqa: E722
    pass

try:
    from .xpu_optimizer_hook import BFp16XPUOptimizerHook

    __all__ += ["BFp16XPUOptimizerHook"]
except:  # noqa: E722
    pass
