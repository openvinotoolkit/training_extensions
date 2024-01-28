"""Utils for demo."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
from pathlib import Path
from typing import Dict, Optional

from .visualizers import (
    ClassificationVisualizer,
    InstanceSegmentationVisualizer,
    ObjectDetectionVisualizer,
    SemanticSegmentationVisualizer,
)


def get_model_path(path: Optional[Path]) -> Path:
    """Get path to model."""
    model_path = path
    if model_path is None:
        model_path = Path(__file__).parent / "openvino.xml"
    if not model_path.exists():
        raise OSError("The path to the model was not found.")

    return model_path


def get_parameters(path: Optional[Path]) -> Dict:
    """Get hyper parameters to creating model."""
    parameters_path = path
    if parameters_path is None:
        parameters_path = Path(__file__).parent / "config.json"
    if not parameters_path.exists():
        raise OSError("The path to the config was not found.")

    with open(parameters_path, encoding="utf8") as file:
        parameters = json.load(file)

    return parameters


def create_visualizer(task_type: str, labels: list, no_show: bool = False, output: Optional[str] = None):
    """Create visualizer according to kind of task."""
    if task_type == "CLASSIFICATION":
        return ClassificationVisualizer(window_name="Result", no_show=no_show, output=output)
    elif task_type == "SEGMENTATION":
        return SemanticSegmentationVisualizer(window_name="Result", labels=labels, no_show=no_show, output=output)
    elif task_type == "INSTANCE_SEGMENTATION":
        return InstanceSegmentationVisualizer(window_name="Result", labels=labels, no_show=no_show, output=output)
    elif task_type == "DETECTION":
        return ObjectDetectionVisualizer(window_name="Result", labels=labels, no_show=no_show, output=output)
    else:
        msg = "Visualizer for f{task_type} is not implemented"
        raise NotImplementedError(msg)
