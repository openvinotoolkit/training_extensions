"""This module contains the base class for non-deep learning image-based tasks. """


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

from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask


class ImageComputerVisionTask(IInferenceTask, metaclass=abc.ABCMeta):
    """
    A base class for a non-deep learning image-based tasks, which can only perform inference.
    This class inherits from ITask and IReporting.

    :example: A cropping task

    >>> class CroppingTask(ImageComputerVisionTask):
    ...     pass
    """
