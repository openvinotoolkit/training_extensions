"""This module implements the OptimizationParameters entity."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
from typing import Callable

from ote_sdk.entities.train_parameters import (
    UpdateProgressCallback,
    default_progress_callback,
)


def default_save_model_callback():
    """
    This is the default save model callback for OptimizationParameters.
    """


@dataclass
class OptimizationParameters:
    """
    Optimization parameters.

    :var resume: Set to ``True`` if optimization must be resume with the optimizer state;
        set to ``False`` to discard the optimizer state and start with fresh optimizer
    :var update_progress: Callback which can be used to provide updates about the progress of a task.
    :var save_model: Callback to notify that the model weights have been changed.
        This callback can be used by the task when temporary weights should be saved (for instance, at the
        end of an epoch). If this callback has been used to save temporary weights, those weights will be
        used to resume optimization if for some reason training was suspended.
    """

    resume: bool = False
    update_progress: UpdateProgressCallback = default_progress_callback
    save_model: Callable[[], None] = default_save_model_callback
