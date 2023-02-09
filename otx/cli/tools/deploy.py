"""Model deployment tool."""

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
from otx.api.entities.model import ModelEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.cli.registry import find_and_parse_model_template
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import read_label_schema, read_model


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    if not os.path.exists("./template.yaml"):
        parser.add_argument("template")
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load model weights from previously saved checkpoint.",
    )
    parser.add_argument(
        "--save-model-to",
        help="Location where openvino.zip will be stored.",
    )

    return parser.parse_args()


def main():
    """Main function that is used for model evaluation."""

    # Parses input arguments.
    args = parse_args()
    if os.path.exists("./template.yaml"):
        template_path = "./template.yaml"
    else:
        template_path = args.template

    # Reads model template file.
    template = find_and_parse_model_template(template_path)

    # Get hyper parameters schema.
    hyper_parameters = template.hyper_parameters.data
    assert hyper_parameters

    # Get classes for Task, ConfigurableParameters and Dataset.
    if not args.load_weights.endswith(".bin") and not args.load_weights.endswith(".xml"):
        raise RuntimeError("Only OpenVINO-exported models are supported.")

    task_class = get_impl_class(template.entrypoints.openvino)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=create(hyper_parameters),
        label_schema=read_label_schema(args.load_weights),
        model_template=template,
    )
    environment.model = read_model(environment.get_model_configuration(), args.load_weights, None)

    task = task_class(task_environment=environment)

    deployed_model = ModelEntity(None, environment.get_model_configuration())

    os.makedirs(args.save_model_to, exist_ok=True)
    task.deploy(deployed_model)
    with open(os.path.join(args.save_model_to, "openvino.zip"), "wb") as write_file:
        write_file.write(deployed_model.exportable_code)

    return dict(retcode=0, template=template.name)


if __name__ == "__main__":
    main()
