"""This module contains the interface class for tasks that can deploy their models. """

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

from ote_sdk.entities.model import ModelEntity


class IDeploymentTask(metaclass=abc.ABCMeta):
    """
    A base interface class for tasks which can deploy their models
    """

    @abc.abstractmethod
    def deploy(self, output_model: ModelEntity) -> None:
        """
        This method defines the interface for deploy.

        :param output_model: Output model
        """
        raise NotImplementedError
