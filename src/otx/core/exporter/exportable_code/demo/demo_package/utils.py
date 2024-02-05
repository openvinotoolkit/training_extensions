# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utils for demo."""

from __future__ import annotations

import json
from pathlib import Path

from .visualizers import (
    BaseVisualizer,
    ClassificationVisualizer,
    InstanceSegmentationVisualizer,
    ObjectDetectionVisualizer,
    SemanticSegmentationVisualizer,
)


def get_model_path(path: Path | None) -> Path:
    """Get path to model."""
    model_path = path
    if model_path is None:
        model_path = Path(__file__).parent / "openvino.xml"
    if not model_path.exists():
        msg = "The path to the model was not found."
        raise OSError(msg)

    return model_path


def get_parameters(path: Path | None) -> dict:
    """Get hyper parameters to creating model."""
    parameters_path = path
    if parameters_path is None:
        parameters_path = Path(__file__).parent / "config.json"
    if not parameters_path.exists():
        msg = "The path to the config was not found."
        raise OSError(msg)

    with Path.open(parameters_path, encoding="utf8") as file:
        return json.load(file)


def create_visualizer(
    task_type: str,
    labels: list,
    no_show: bool = False,
    output: str = "./outputs",
) -> BaseVisualizer | None:
    """Create visualizer according to kind of task."""
    if task_type == "CLASSIFICATION":
        return ClassificationVisualizer(window_name="Result", no_show=no_show, output=output)
    if task_type == "SEGMENTATION":
        return SemanticSegmentationVisualizer(window_name="Result", labels=labels, no_show=no_show, output=output)
    if task_type == "INSTANCE_SEGMENTATION":
        return InstanceSegmentationVisualizer(window_name="Result", labels=labels, no_show=no_show, output=output)
    if task_type == "DETECTION":
        return ObjectDetectionVisualizer(window_name="Result", labels=labels, no_show=no_show, output=output)
    msg = "Visualizer for f{task_type} is not implemented"
    raise NotImplementedError(msg)
