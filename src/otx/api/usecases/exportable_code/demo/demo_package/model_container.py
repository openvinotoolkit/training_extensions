"""ModelContainer class used for loading the model in the model wrapper."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from openvino.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_api.models import ImageModel, Model
from openvino.model_api.tilers import DetectionTiler, InstanceSegmentationTiler

from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model_template import TaskType
from otx.api.serialization.label_mapper import LabelSchemaMapper
from otx.api.utils.detection_utils import detection2array
from otx.api.utils.tiler import Tiler

from .utils import get_model_path, get_parameters


class ModelContainer:
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

        self._labels = LabelSchemaMapper.backward(self.parameters["model_parameters"]["labels"])
        self._task_type = TaskType[self.parameters["converter_type"]]

        self.segm = bool(
            self._task_type is TaskType.ROTATED_DETECTION or self._task_type is TaskType.INSTANCE_SEGMENTATION
        )

        # labels for modelAPI wrappers can be empty, because unused in pre- and postprocessing
        self.model_parameters = self.parameters["model_parameters"]

        if self._task_type in (
            TaskType.ANOMALY_CLASSIFICATION,
            TaskType.ANOMALY_DETECTION,
            TaskType.ANOMALY_SEGMENTATION,
        ):
            # The anomaly task requires non-empty labels.
            # modelapi_labels key is used as a workaround as labels key is used for labels in OTX SDK format
            self.model_parameters["labels"] = (
                self.model_parameters.pop("modelapi_labels")
                if "modelapi_labels" in self.model_parameters
                else ["Normal", "Anomaly"]
            )
        else:
            # model already contains correct labels
            self.model_parameters.pop("labels")

        self._initialize_wrapper()
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

        classifier = None
        if self.parameters["tiling_parameters"].get("enable_tile_classifier", False):
            adapter = OpenvinoAdapter(create_core(), get_model_path(model_dir / "tile_classifier.xml"), device=device)
            classifier = ImageModel(inference_adapter=adapter, configuration={}, preload=True)

        tiler_config = {
            "tile_size": int(
                self.parameters["tiling_parameters"]["tile_size"]
                * self.parameters["tiling_parameters"]["tile_ir_scale_factor"]
            ),
            "tiles_overlap": self.parameters["tiling_parameters"]["tile_overlap"]
            / self.parameters["tiling_parameters"]["tile_ir_scale_factor"],
            "max_pred_number": self.parameters["tiling_parameters"]["tile_max_number"],
        }

        if self.segm:
            return InstanceSegmentationTiler(self.core_model, tiler_config, tile_classifier_model=classifier)
        else:
            return DetectionTiler(self.core_model, tiler_config)

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
            print("Using model wrapper from ModelAPI")

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

        # MaskRCNN returns tuple so no need to process
        if self._task_type == TaskType.DETECTION:
            predictions = detection2array(predictions.objects)
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
