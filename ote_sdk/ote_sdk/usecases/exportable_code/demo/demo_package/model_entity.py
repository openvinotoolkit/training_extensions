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
from openvino.model_zoo.model_api.pipelines import get_user_config

from ote_sdk.serialization.label_mapper import LabelSchemaMapper

from .utils import get_model_path, get_parameters


class ModelEntity:
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

        plugin_config = get_user_config("CPU", "", None)
        model_adapter = OpenvinoAdapter(
            create_core(),
            get_model_path(model_dir / "model.xml"),
            plugin_config=plugin_config,
        )

        self._initialize_wrapper(model_dir.parent.resolve() / "python" / "model.py")
        self.core_model = Model.create_model(
            self.parameters["type_of_model"],
            model_adapter,
            self.model_parameters,
            preload=True,
        )

    @staticmethod
    def _initialize_wrapper(path_to_wrapper: Path):
        if path_to_wrapper:
            if not path_to_wrapper.exists():
                raise IOError("The path to the model.py was not found.")

            spec = importlib.util.spec_from_file_location("model", path_to_wrapper)  # type: ignore
            model = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(model)
        else:
            print("Using model wrapper from Open Model Zoo ModelAPI")
