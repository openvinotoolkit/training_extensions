"""ModelContainer class used for loading the model in the model wrapper."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from otx.api.usecases.model_api.adapters import OpenvinoAdapter, create_core
from otx.api.usecases.model_api.models import Model

from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model_template import TaskType
from otx.api.serialization.label_mapper import LabelSchemaMapper

from .utils import get_model_path, get_parameters


class ModelContainer:
    """Class for storing the model wrapper based on Model API and needed parameters of model.

    Args:
        model_dir (Path): path to model directory
    """

    def __init__(self, model_dir: Path) -> None:
        self.parameters = get_parameters(model_dir / "config.json")
        self._labels = LabelSchemaMapper.backward(self.parameters["model_parameters"]["labels"])
        self._task_type = TaskType[self.parameters["converter_type"]]

        # labels for modelAPI wrappers can be empty, because unused in pre- and postprocessing
        self.model_parameters = self.parameters["model_parameters"]
        self.model_parameters["labels"] = []

        model_adapter = OpenvinoAdapter(create_core(), get_model_path(model_dir / "model.xml"))

        self._initialize_wrapper()
        self.core_model = Model.create_model(
            self.parameters["type_of_model"],
            model_adapter,
            self.model_parameters,
            preload=True,
        )

    @property
    def task_type(self) -> TaskType:
        """Task type property."""
        return self._task_type

    @property
    def labels(self) -> LabelSchemaEntity:
        """Labels property."""
        return self._labels

    @staticmethod
    def _initialize_wrapper() -> None:
        """Load the model class."""
        try:
            importlib.import_module("model_wrappers")
        except ModuleNotFoundError:
            print("Using model wrapper from Open Model Zoo ModelAPI")

    def __call__(self, input_data: np.ndarray) -> Tuple[Any, dict]:
        """Returns the output of the model.

        # TODO possibly unused. Remove?

        Args:
            input_data (np.ndarray): Input image/video data.

        Returns:
            Tuple[Any, dict]: Model predictions.
        """
        return self.core_model(input_data)
