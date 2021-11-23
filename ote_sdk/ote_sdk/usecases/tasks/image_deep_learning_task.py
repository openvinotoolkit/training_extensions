"""This module contains the base class for deep learning image-based tasks. """


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

import abc

from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask


class ImageDeepLearningTask(IInferenceTask, ITrainingTask, IEvaluationTask, metaclass=abc.ABCMeta):
    """
    A base class for a deep learning image-based tasks.
    This class inherits from ITask, ITraining, IComputesPerformance and IReporting.

    :example: A YOLO detection task.

    >>> class YOLODetection(ImageDeepLearningTask):
    ...     pass
    """
