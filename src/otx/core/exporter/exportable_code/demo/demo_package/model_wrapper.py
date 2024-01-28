"""ModelContainer class used for loading the model in the model wrapper."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from openvino.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_api.models import Model
from openvino.model_api.tilers import DetectionTiler, InstanceSegmentationTiler

from .utils import get_model_path, get_parameters


class TaskType(str, Enum):
    """OTX task type definition."""

    CLASSIFICATION = "CLASSIFICATION"
    DETECTION = "DETECTION"
    INSTANCE_SEGMENTATION = "INSTANCE_SEGMENTATION"
    SEGMENTATION = "SEGMENTATION"


class ModelWrapper:
    """Class for storing the model wrapper based on Model API and needed parameters of model.

    Args:
        model_dir (Path): path to model directory
    """

    def __init__(self, model_dir: Path, device="CPU") -> None:
        model_adapter = OpenvinoAdapter(create_core(), get_model_path(model_dir / "model.xml"), device=device)

        try:
            config_data = model_adapter.model.get_rt_info(["otx_config"])
            if type(config_data) != str:
                # OV 2023.0 return OVAny which needs to be casted with astype()
                config_data = config_data.astype(str)
            self.parameters = json.loads(config_data)
        except RuntimeError:
            self.parameters = get_parameters(model_dir / "config.json")
        self._labels = self.parameters["model_parameters"]["labels"]
        self._task_type = TaskType[self.parameters["converter_type"].upper()]

        self.segm = bool(
            self._task_type is TaskType.INSTANCE_SEGMENTATION,
        )

        # labels for modelAPI wrappers can be empty, because unused in pre- and postprocessing
        self.model_parameters = self.parameters["model_parameters"]

        # model already contains correct labels
        self.model_parameters.pop("labels")

        self.core_model = Model.create_model(
            model_adapter,
            self.parameters["type_of_model"],
            self.model_parameters,
            preload=True,
        )
        self.tiler = self.setup_tiler(model_dir, device)

    def setup_tiler(self, model_dir, device) -> Optional[Union[DetectionTiler, InstanceSegmentationTiler]]:
        """Setup tiler for model.

        Args:
            model_dir (str): model directory
            device (str): device to run model on
        Returns:
            Optional: Tiler object or None
        """
        if not self.parameters.get("tiling_parameters") or not self.parameters["tiling_parameters"]["enable_tiling"]:
            return None

        msg = "Tiling has not implemented yet"
        raise NotImplementedError(msg)

    @property
    def task_type(self) -> TaskType:
        """Task type property."""
        return self._task_type

    @property
    def labels(self) -> dict:
        """Labels property."""
        return self._labels

    def infer(self, frame):
        """Infer with original image.

        Args:
            frame (np.ndarray): image
        Returns:
            annotation_scene (AnnotationScene): prediction
            frame_meta (Dict): dict with original shape
        """
        # getting result include preprocessing, infer, postprocessing for sync infer
        predictions = self.core_model(frame)
        frame_meta = {"original_shape": frame.shape}

        return predictions, frame_meta

    def infer_tile(self, frame):
        """Infer by patching full image to tiles.

        Args:
            frame (np.ndarray): image
        Returns:
            annotation_scene (AnnotationScene): prediction
            frame_meta (Dict): dict with original shape
        """
        detections = self.tiler(frame)
        return detections, {"original_shape": frame.shape}

    def __call__(self, input_data: np.ndarray) -> Tuple[Any, dict]:
        """Infer entry wrapper."""
        if self.tiler:
            return self.infer_tile(input_data)
        return self.infer(input_data)
