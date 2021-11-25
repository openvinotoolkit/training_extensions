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
import os
from pathlib import Path
from openvino.model_zoo.model_api import models, pipelines
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter
from ote_sdk.entities.label import Domain
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import create_converter

def get_model_path(path):
    model_path = path
    if model_path is None:
        model_path = Path(__file__).parent / 'model.xml'
        if not os.path.exists(model_path):
            raise IOError("The path to the model was not found.")

    return model_path

def get_parameters(path):
    parameters_path = path
    if parameters_path is None:
        parameters_path = Path(__file__).parent / 'config.json'
        if not os.path.exists(parameters_path):
            raise IOError("The path to the config was not found.")

    with open(parameters_path, 'r') as f:
        parameters = json.load(f)

    return parameters

def create_model(infer_parameters, model_path=None, config_file=None):
    plugin_config = pipelines.get_user_config(infer_parameters.device, infer_parameters.streams, infer_parameters.threads)
    model_adapter = OpenvinoAdapter(create_core(), get_model_path(model_path), device=infer_parameters.device,
                                    plugin_config=plugin_config, max_num_requests=infer_parameters.infer_requests)
    parameters = get_parameters(config_file)
    try:
        importlib.import_module('.model', parameters['name_of_model'].lower())
    except BaseException:
        print("Using model wrapper from Open Model Zoo ModelAPI")
    model = models.Model.create_model(parameters['type_of_model'], model_adapter, parameters['model_parameters'])
    model.load()

    return model

def create_output_converter(labels, config_file=None):
    parameters = get_parameters(config_file)
    type = Domain[parameters['converter_type']]
    return create_converter(type, labels)
