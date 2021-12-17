"""
Model deployment tool.
"""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import os

from ote_sdk.configuration.helper import create
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.model import ModelEntity

from ote_cli.registry import find_and_parse_model_template
from ote_cli.utils.config import override_parameters
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.io import read_label_schema, read_model
from ote_cli.utils.parser import gen_params_dict_from_args
from ote_cli.utils.io import read_label_schema, save_model_data


def parse_args():
    """
    Parses command line arguments.
    """

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("template")
    parsed, _ = pre_parser.parse_known_args()
    # Load template.yaml file.
    template = find_and_parse_model_template(parsed.template)
    # Get hyper parameters schema.
    hyper_parameters = template.hyper_parameters.data
    assert hyper_parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("template")
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load only weights from previously saved checkpoint",
    )
    parser.add_argument(
        "--save-model-to",
        help="Path to zip file where ",
    )

    return parser.parse_args(), template, hyper_parameters


def main():
    """
    Main function that is used for model evaluation.
    """

    # Dynamically create an argument parser based on override parameters.
    args, template, hyper_parameters = parse_args()
    # Get new values from user's input.
    updated_hyper_parameters = gen_params_dict_from_args(args)
    # Override overridden parameters by user's values.
    override_parameters(updated_hyper_parameters, hyper_parameters)

    hyper_parameters = create(hyper_parameters)

    # Get classes for Task, ConfigurableParameters and Dataset.
    if not args.load_weights.endswith(".bin") and not args.load_weights.endswith(".xml"):
        raise RuntimeError("Only OpenVINO-exported models are supported.")
    
    task_class = get_impl_class(template.entrypoints.openvino)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=read_label_schema(
            os.path.join(os.path.dirname(args.load_weights), "label_schema.json")
        ),
        model_template=template,
    )
    environment.model = read_model(environment.get_model_configuration(), args.load_weights, None)
    
    task = task_class(task_environment=environment)
    
    deployed_model = ModelEntity(
        None, environment.get_model_configuration()
    )
    
    os.makedirs(args.save_model_to, exist_ok=True)
    task.deploy(deployed_model)
    with open(os.path.join(args.save_model_to, "openvino.zip"), "wb") as write_file:
        write_file.write(deployed_model.exportable_code)
