"""This module contains the interface class for tasks that can export their models. """

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

from ote_sdk.entities.model import ModelEntity


class ExportType(Enum):
    """
    Represent the type of export format available through this interface.
    """

    OPENVINO = auto()


class IExportTask(metaclass=abc.ABCMeta):
    """
    A base interface class for tasks which can export their models
    """

    @abc.abstractmethod
    def export(self, export_type: ExportType, output_model: ModelEntity):
        """
        This method defines the interface for export.

        :param export_type: The type of optimization
        :param output_model: Output model
        """
        raise NotImplementedError
