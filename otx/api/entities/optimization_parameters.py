"""This module implements the OptimizationParameters entity."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
from typing import Any, Callable, Optional


# pylint: disable=unused-argument
def default_progress_callback(progress: int, score: Optional[float] = None):
    """This is the default progress callback for OptimizationParameters."""


def default_save_model_callback():
    """This is the default save model callback for OptimizationParameters."""


@dataclass
class OptimizationParameters:
    """Optimization parameters.

    resume (bool): Set to ``True`` if optimization must be resume with the optimizer state;
        set to ``False`` to discard the optimizer state and start with fresh optimizer
    update_progress (Callable[int], None): Callback which can be used to provide updates about the progress of a task.
    save_model (Callable[List, None]): Callback to notify that the model weights have been changed.
        This callback can be used by the task when temporary weights should be saved (for instance, at the
        end of an epoch). If this callback has been used to save temporary weights, those weights will be
        used to resume optimization if for some reason training was suspended.
    """

    resume: bool = False
    update_progress: Callable[[int, Optional[float]], Any] = default_progress_callback
    save_model: Callable[[], None] = default_save_model_callback
