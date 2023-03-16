"""This module implements the TrainingParameters entity."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol


class UpdateProgressCallback(Protocol):
    """UpdateProgressCallback protocol.

    Used as a replacement of Callable[] type since Callable doesn’t handle default parameters like
    `score: Optional[float] = None`
    """

    def __call__(self, progress: float, score: Optional[float] = None):
        """Callback to provide updates about the progress of a task.

        It is recommended to call this function at least once per epoch.
        However, the exact frequency is left to the task implementer.

        An optional `score` can also be passed. If specified, this score can be used by HPO
        to monitor the improvement of the task.

        Args:
            progress: Progress as a percentage
            score: Optional validation score
        """


# pylint: disable=unused-argument
def default_progress_callback(progress: float, score: Optional[float] = None):
    """Default progress callback. It is a placeholder (does nothing) and is used in empty TrainParameters."""


def default_save_model_callback():
    """Default save model callback. It is a placeholder (does nothing) and is used in empty TrainParameters."""


@dataclass
class TrainParameters:
    """Train parameters.

    Attributes:
        resume: Set to ``True`` if training must be resume with the
            optimizer state; set to ``False`` to discard the optimizer
            state and start with fresh optimizer
        update_progress: Callback which can be used to provide updates
            about the progress of a task.
        save_model: Callback to notify that the model weights have been
            changed. This callback can be used by the task when
            temporary weights should be saved (for instance, at the end
            of an epoch). If this callback has been used to save
            temporary weights, those weights will be used to resume
            training if for some reason training was suspended.
    """

    resume: bool = False
    update_progress: Callable[[int, Optional[float]], Any] = default_progress_callback
    save_model: Callable[[], None] = default_save_model_callback
