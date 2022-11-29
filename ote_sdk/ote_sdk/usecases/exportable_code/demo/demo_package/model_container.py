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
from ote_sdk.entities.model_template import TaskType
from ote_sdk.serialization.label_mapper import LabelSchemaMapper
from ote_sdk.utils import Tiler
from ote_sdk.utils.detection_utils import detection2array

from .utils import get_model_path, get_parameters


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
        self._task_type = TaskType[self.parameters["converter_type"]]

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

        self.tiler = self.setup_tiler()

    def setup_tiler(self):
        """Setup tiler

        Returns:
            Tiler: tiler module
        """
        if (
            not self.parameters.get("tiling_parameters")
            or not self.parameters["tiling_parameters"]["enable_tiling"]["value"]
        ):
            return None

        segm = False
        if (
            self._task_type is TaskType.ROTATED_DETECTION
            or self._task_type is TaskType.INSTANCE_SEGMENTATION
        ):
            segm = True
        tile_size = self.parameters["tiling_parameters"]["tile_size"]["value"]
        tile_overlap = self.parameters["tiling_parameters"]["tile_overlap"]["value"]
        max_number = self.parameters["tiling_parameters"]["tile_max_number"]["value"]
        tiler = Tiler(tile_size, tile_overlap, max_number, self.core_model, segm)
        return tiler

    @property
    def task_type(self) -> TaskType:
        """
        Task type property
        """
        return self._task_type

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

    def infer(self, frame):
        """Infer with original image
        Args:
            frame (np.ndarray): image
        Returns:
            annotation_scene (AnnotationScene): prediction
            frame_meta (Dict): dict with original shape
        """
        # getting result include preprocessing, infer, postprocessing for sync infer
        predictions, frame_meta = self.core_model(frame)
        predictions = detection2array(predictions)
        return predictions, frame_meta

    def infer_tile(self, frame):
        """Infer by patching full image to tiles
        Args:
            frame (np.ndarray): image
        Returns:
            annotation_scene (AnnotationScene): prediction
            frame_meta (Dict): dict with original shape
        """

        detections, _ = self.tiler.predict(frame)
        return detections, {"original_shape": frame.shape}

    def __call__(self, input_data: np.ndarray) -> Tuple[Any, dict]:
        if self.tiler:
            return self.infer_tile(input_data)
        return self.infer(input_data)
