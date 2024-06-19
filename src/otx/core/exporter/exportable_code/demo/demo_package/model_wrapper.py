# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""ModelContainer class used for loading the model in the model wrapper."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

from model_api.adapters import OpenvinoAdapter, create_core
from model_api.models import Model

from .utils import get_model_path, get_parameters

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from model_api.tilers import DetectionTiler, InstanceSegmentationTiler


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

    def __init__(self, model_dir: Path, device: str = "CPU") -> None:
        model_adapter = OpenvinoAdapter(create_core(), get_model_path(model_dir / "model.xml"), device=device)
        if not (model_dir / "config.json").exists():
            msg = "config.json doesn't exist in the model directory."
            raise RuntimeError(msg)
        self.parameters = get_parameters(model_dir / "config.json")
        self._labels = self.parameters["model_parameters"]["labels"]
        self._task_type = TaskType[self.parameters["task_type"].upper()]

        # labels for modelAPI wrappers can be empty, because unused in pre- and postprocessing
        self.model_parameters = self.parameters["model_parameters"]

        # model already contains correct labels
        self.model_parameters.pop("labels")

        self.core_model = Model.create_model(
            model_adapter,
            self.parameters["model_type"],
            self.model_parameters,
            preload=True,
        )
        self.tiler = self.setup_tiler(model_dir, device)

    def setup_tiler(
        self,
        model_dir: Path,
        device: str,
    ) -> DetectionTiler | InstanceSegmentationTiler | None:
        """Set up tiler for model.

        Args:
            model_dir (str): model directory
            device (str): device to run model on
        Returns:
            Optional: type of tiler or None
        """
        if not self.parameters.get("tiling_parameters") or not self.parameters["tiling_parameters"]["enable_tiling"]:
            return None

        msg = "Tiling has not been implemented yet"
        raise NotImplementedError(msg)

    @property
    def task_type(self) -> TaskType:
        """Task type property."""
        return self._task_type

    @property
    def labels(self) -> dict:
        """Labels property."""
        return self._labels

    def infer(self, frame: np.ndarray) -> tuple[NamedTuple, dict]:
        """Infer with original image.

        Args:
            frame: np.ndarray, input image
        Returns:
            predictions: NamedTuple, prediction
            frame_meta: Dict, dict with original shape
        """
        # getting result include preprocessing, infer, postprocessing for sync infer
        predictions = self.core_model(frame)
        frame_meta = {"original_shape": frame.shape}

        return predictions, frame_meta

    def infer_tile(self, frame: np.ndarray) -> tuple[NamedTuple, dict]:
        """Infer by patching full image to tiles.

        Args:
            frame: np.ndarray - input image
        Returns:
            Tuple[NamedTuple, Dict]: prediction and original shape
        """
        if self.tiler is None:
            msg = "Tiler is not set"
            raise RuntimeError(msg)
        detections = self.tiler(frame)
        return detections, {"original_shape": frame.shape}

    def __call__(self, input_data: np.ndarray) -> tuple[Any, dict]:
        """Call the ModelWrapper class.

        Args:
            input_data (np.ndarray): The input image.

        Returns:
            Tuple[Any, dict]: A tuple containing predictions and the meta information.
        """
        if self.tiler is not None:
            return self.infer_tile(input_data)
        return self.infer(input_data)
