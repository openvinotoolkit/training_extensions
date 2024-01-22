"""Utils for demo."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
from pathlib import Path
from typing import Dict, Optional

from .model_wrapper import TaskType
from .visualizers import FakeVisualizer, Visualizer


def get_model_path(path: Optional[Path]) -> Path:
    """Get path to model."""
    model_path = path
    if model_path is None:
        model_path = Path(__file__).parent / "model.xml"
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


def create_visualizer(task_type: TaskType, no_show: bool = False, output: Optional[str] = None):
    """Create visualizer according to kind of task."""
    # TODO: use anomaly-specific visualizer for anomaly tasks
    if task_type == "Classification":
        return Visualizer(window_name="Result", no_show=no_show, output=output)
    # TODO: add task specific visualizers when implemented
    return FakeVisualizer(window_name="Result", no_show=no_show, output=output)
