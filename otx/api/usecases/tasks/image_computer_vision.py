"""This module contains the base class for non-deep learning image-based tasks."""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc

from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask


class ImageComputerVisionTask(IInferenceTask, metaclass=abc.ABCMeta):
    """A base class for a non-deep learning image-based tasks, which can only perform inference.

    This class inherits from ITask and IReporting.

    Example:
        A cropping task

    >>> class CroppingTask(ImageComputerVisionTask):
    ...     pass
    """
