"""
Utils for demo
"""
# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

import importlib
import json
from pathlib import Path
from typing import List, Optional, Union

from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_zoo.model_api.models import Model

from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import (
    create_converter,
)


def get_model_path(path: Optional[Path]) -> Path:
    """
    Get path to model
    """
    model_path = path
    if model_path is None:
        model_path = Path(__file__).parent / "model.xml"
        if not model_path.exists():
            raise IOError("The path to the model was not found.")

    return model_path


def get_parameters(path: Optional[Path]) -> dict:
    """
    Get hyper parameters to creating model
    """
    parameters_path = path
    if parameters_path is None:
        parameters_path = Path(__file__).parent / "config.json"
    if not parameters_path.exists():
        raise IOError("The path to the config was not found.")

    with open(parameters_path, "r", encoding="utf8") as file:
        parameters = json.load(file)

    return parameters


def create_model(model_path: Path, config_file: Optional[Path] = None) -> Model:
    """
    Create model using ModelAPI factory
    """

    model_adapter = OpenvinoAdapter(create_core(), get_model_path(model_path))
    parameters = get_parameters(config_file)
    try:
        importlib.import_module(".model", "demo_package")
    except ImportError:
        print("Using model wrapper from Open Model Zoo ModelAPI")
    model = Model.create_model(
        parameters["type_of_model"],
        model_adapter,
        parameters["model_parameters"],
        preload=True,
    )

    return model


def create_output_converter(
    labels: List[Union[str, LabelEntity]], config_file: Path = None
):
    """
    Create annotation converter according to kind of task
    """
    parameters = get_parameters(config_file)
    converter_type = Domain[parameters["converter_type"]]
    return create_converter(converter_type, labels)
