"""This module contains the interface class for tasks that can optimize their models. """


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
from enum import Enum, auto
from typing import Optional

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.optimization_parameters import OptimizationParameters


class OptimizationType(Enum):
    """
    This class enumerates the OPENVINO optimization types.
    """

    POT = auto()
    NNCF = auto()


class IOptimizationTask(metaclass=abc.ABCMeta):
    """
    A base interface class for tasks which can optimize their models
    """

    @abc.abstractmethod
    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters],
    ):
        """
        This method defines the interface for optimization.

        :param optimization_type: The type of optimization
        :param dataset: Optional dataset which may be used as part of the optimization process
        :param output_model: Output model
        :param optimization_parameters: Additional optimization parameters
        """
        raise NotImplementedError
