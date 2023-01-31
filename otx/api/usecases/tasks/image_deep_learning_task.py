"""This module contains the base class for deep learning image-based tasks."""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc

from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask


class ImageDeepLearningTask(IInferenceTask, ITrainingTask, IEvaluationTask, metaclass=abc.ABCMeta):
    """A base class for a deep learning image-based tasks.

    This class inherits from ITask, ITraining, IComputesPerformance and IReporting.

    Example:
        A YOLO detection task.

        >>> class YOLODetection(ImageDeepLearningTask):
        ...     pass
    """
