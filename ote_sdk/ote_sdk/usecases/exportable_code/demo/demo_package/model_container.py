"""
ModelContainer
"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_zoo.model_api.models import Model

from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.serialization.label_mapper import LabelSchemaMapper

from .utils import get_model_path, get_parameters

GLOBAL_TASK_TYPES = ["CLASSIFICATION", "ANOMALY_CLASSIFICATION"]


class ModelContainer:
    """
    Class for storing the model wrapper based on Model API and needed parameters of model

    Args:
        model_dir: path to model directory
    """

    def __init__(self, model_dir: Path) -> None:
        self.parameters = get_parameters(model_dir / "config.json")
        self._labels = LabelSchemaMapper.backward(
            self.parameters["model_parameters"]["labels"]
        )
        self._task_type = self.parameters["converter_type"]

        # labels for modelAPI wrappers can be empty, because unused in pre- and postprocessing
        self.model_parameters = self.parameters["model_parameters"]
        self.model_parameters["labels"] = []

        model_adapter = OpenvinoAdapter(
            create_core(), get_model_path(model_dir / "model.xml")
        )

        self._initialize_wrapper()
        self.core_model = Model.create_model(
            self.parameters["type_of_model"],
            model_adapter,
            self.model_parameters,
            preload=True,
        )

    @property
    def task_type(self) -> str:
        """
        Task type property
        """
        return self._task_type

    @property
    def is_global(self) -> bool:
        """
        Return True if the task produces global labels, False otherwise
        """
        return self._task_type in GLOBAL_TASK_TYPES

    @property
    def labels(self) -> LabelSchemaEntity:
        """
        Labels property
        """
        return self._labels

    @staticmethod
    def _initialize_wrapper() -> None:
        try:
            importlib.import_module("model_wrappers")
        except ModuleNotFoundError:
            print("Using model wrapper from Open Model Zoo ModelAPI")

    def __call__(self, input_data: np.ndarray) -> Tuple[Any, dict]:
        return self.core_model(input_data)
