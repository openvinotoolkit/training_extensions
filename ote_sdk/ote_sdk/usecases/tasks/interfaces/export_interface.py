"""This module contains the interface class for tasks that can export their models. """

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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
