"""This module contains the interface class for tasks that can export their models."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from enum import Enum, auto

from otx.api.entities.model import ModelEntity, ModelPrecision


class ExportType(Enum):
    """Represent the type of export format available through this interface."""

    OPENVINO = auto()
    ONNX = auto()


class IExportTask(metaclass=abc.ABCMeta):
    """A base interface class for tasks which can export their models."""

    @abc.abstractmethod
    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision,
        dump_features: bool,
    ):
        """This method defines the interface for export.

        Args:
            export_type (ExportType): The type of optimization.
            output_model (ModelEntity): The output model entity.
            precision (ModelPrecision): The precision of the ouptut model.
            dump_features (bool): Flag to return "feature_vector" and "saliency_map".
        """
        raise NotImplementedError
