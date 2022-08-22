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

from otx.api.configuration.helper import create
from otx.api.entities.model import ModelEntity, ModelOptimizationType
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.cli.registry import find_and_parse_model_template
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import read_binary, read_label_schema, save_model_data
from otx.cli.utils.nncf import is_checkpoint_nncf


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
    is_nncf = is_checkpoint_nncf(args.load_weights)
    task_class = get_impl_class(
        template.entrypoints.nncf if is_nncf else template.entrypoints.base
    )

    # Get hyper parameters schema.
    hyper_parameters = create(template.hyper_parameters.data)
    assert hyper_parameters

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=read_label_schema(args.load_weights),
        model_template=template,
    )

    model_adapters = {"weights.pth": ModelAdapter(read_binary(args.load_weights))}
    model = ModelEntity(
        configuration=environment.get_model_configuration(),
        model_adapters=model_adapters,
        train_dataset=None,
        optimization_type=ModelOptimizationType.NNCF
        if is_nncf
        else ModelOptimizationType.NONE,
    )
    environment.model = model

    task = task_class(task_environment=environment)

    exported_model = ModelEntity(None, environment.get_model_configuration())

    task.export(ExportType.OPENVINO, exported_model)

    os.makedirs(args.save_model_to, exist_ok=True)
    save_model_data(exported_model, args.save_model_to)


if __name__ == "__main__":
    main()
