"""
ModelEntity
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
from pathlib import Path

from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_zoo.model_api.models import Model

from ote_sdk.serialization.label_mapper import LabelSchemaMapper

from .utils import get_model_path, get_parameters


class ModelContainer:
    """
    Class for storing the model wrapper based on Model API and needed parameters of model

    Args:
        model_dir: path to model directory
    """

    def __init__(self, model_dir: Path) -> None:
        self.parameters = get_parameters(model_dir / "config.json")
        self.labels = LabelSchemaMapper.backward(
            self.parameters["model_parameters"]["labels"]
        )
        self.task_type = self.parameters["converter_type"]

        # labels for modelAPI wrappers can be empty, because unused in pre- and postprocessing
        self.model_parameters = self.parameters["model_parameters"]
        self.model_parameters["labels"] = []

        model_adapter = OpenvinoAdapter(
            create_core(), get_model_path(model_dir / "model.xml")
        )

        self._initialize_wrapper(model_dir.parent.resolve())
        self.core_model = Model.create_model(
            self.parameters["type_of_model"],
            model_adapter,
            self.model_parameters,
            preload=True,
        )

    @staticmethod
    def _initialize_wrapper(wrapper_dir: Path):
        if wrapper_dir:
            if not wrapper_dir.exists():
                raise IOError("The path to wrappers was not found.")

            importlib.import_module("model_wrappers")
        else:
            print("Using model wrapper from Open Model Zoo ModelAPI")

    def __call__(self, input_data):
        return self.core_model(input_data)
