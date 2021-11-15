"""This module implements the AnalyseParameters entity"""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.


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
