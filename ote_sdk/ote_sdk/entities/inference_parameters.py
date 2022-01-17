"""This module implements the AnalyseParameters entity"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from dataclasses import dataclass
from typing import Callable


def default_progress_callback(_: int):
    """
    Default progress callback. It is a placeholder (does nothing) and is used in empty InferenceParameters.
    """


@dataclass
class InferenceParameters:
    """
    Inference parameters

    :var is_evaluation: Set to ``True`` if the output dataset is intended to be used for evaluation purposes.
        In this scenario, any postprocessing filtering (such as thresholding and NMS) should be disabled to avoid
        interfering with algorithms such as NMS.
    :var update_progress: Callback which can be used to provide updates about the progress of a task.
    """

    is_evaluation: bool = False
    update_progress: Callable[[int], None] = default_progress_callback
