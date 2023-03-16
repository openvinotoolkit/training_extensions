# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# flake8: noqa
from otx import MMDET_AVAILABLE

from . import (
    adaptive_training_hooks,
    composed_dataloaders_hook,
    early_stopping_hook,
    fp16_sam_optimizer_hook,
    logger_replace_hook,
    model_ema_hook,
    model_ema_v2_hook,
    recording_forward_hooks,
    save_initial_weight_hook,
    task_adapt_hook,
    unbiased_teacher_hook,
    workflow_hooks,
)

if MMDET_AVAILABLE:
    from . import det_saliency_map_hook
