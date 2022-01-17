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
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.task_environment import TaskEnvironment

from ote_cli.registry import find_and_parse_model_template
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.io import read_label_schema, read_model


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("template")
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load only weights from previously saved checkpoint.",
    )
    parser.add_argument(
        "--save-model-to",
        help="Location where openvino.zip will be stored.",
    )

    return parser.parse_args()


def main():
    """
    Main function that is used for model evaluation.
    """

    # Parses input arguments.
    args = parse_args()

    # Reads model template file.
    template = find_and_parse_model_template(args.template)

    # Get hyper parameters schema.
    hyper_parameters = template.hyper_parameters.data
    assert hyper_parameters

    # Get classes for Task, ConfigurableParameters and Dataset.
    if not args.load_weights.endswith(".bin") and not args.load_weights.endswith(
        ".xml"
    ):
        raise RuntimeError("Only OpenVINO-exported models are supported.")

    task_class = get_impl_class(template.entrypoints.openvino)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=create(hyper_parameters),
        label_schema=read_label_schema(
            os.path.join(os.path.dirname(args.load_weights), "label_schema.json")
        ),
        model_template=template,
    )
    environment.model = read_model(
        environment.get_model_configuration(), args.load_weights, None
    )

    task = task_class(task_environment=environment)

    deployed_model = ModelEntity(None, environment.get_model_configuration())

    os.makedirs(args.save_model_to, exist_ok=True)
    task.deploy(deployed_model)
    with open(os.path.join(args.save_model_to, "openvino.zip"), "wb") as write_file:
        write_file.write(deployed_model.exportable_code)


if __name__ == "__main__":
    main()
