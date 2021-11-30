"""
Model exporting tool.
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
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType

from ote_cli.registry import find_and_parse_model_template
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.loading import load_model_weights, read_label_schema


def parse_args():
    """
    Parses command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("template")
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load only weights from previously saved checkpoint",
    )
    parser.add_argument(
        "--save-model-to",
        required="True",
        help="Location where exported model will be stored.",
    )

    return parser.parse_args()


def main():
    """
    Main function that is used for model exporting.
    """

    args = parse_args()

    # Load template.yaml file.
    template = find_and_parse_model_template(args.template)

    # Get class for Task.
    task_class = get_impl_class(template.entrypoints.base)

    # Get hyper parameters schema.
    hyper_parameters = create(template.hyper_parameters.data)
    assert hyper_parameters

    model_bytes = load_model_weights(args.load_weights)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=read_label_schema(model_bytes),
        model_template=template,
    )

    model_adapters = {"weights.pth": ModelAdapter(model_bytes)}
    model = ModelEntity(
        configuration=environment.get_model_configuration(),
        model_adapters=model_adapters,
        train_dataset=None,
    )
    environment.model = model

    task = task_class(task_environment=environment)

    exported_model = ModelEntity(
        None, environment.get_model_configuration(), model_status=ModelStatus.NOT_READY
    )

    task.export(ExportType.OPENVINO, exported_model)

    os.makedirs(args.save_model_to, exist_ok=True)

    with open(os.path.join(args.save_model_to, "model.bin"), "wb") as write_file:
        write_file.write(exported_model.get_data("openvino.bin"))

    with open(
        os.path.join(args.save_model_to, "model.xml"), "w", encoding="UTF-8"
    ) as write_file:
        write_file.write(exported_model.get_data("openvino.xml").decode())
