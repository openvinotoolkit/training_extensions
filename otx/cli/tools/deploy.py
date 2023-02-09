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

import os
from pathlib import Path

from otx.api.configuration.helper import create
from otx.api.entities.model import ModelEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.cli.manager import ConfigManager
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import read_label_schema, read_model
from otx.cli.utils.parser import get_parser_and_hprams_data


def get_args():
    """Parses command line arguments."""
    parser, _, _ = get_parser_and_hprams_data()

    parser.add_argument(
        "--load-weights",
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
    args = get_args()
    config_manager = ConfigManager(args, mode="deploy")
    # Auto-Configuration for model template
    config_manager.configure_template()

    # Reads model template file.
    template = config_manager.template

    # Get hyper parameters schema.
    hyper_parameters = template.hyper_parameters.data
    assert hyper_parameters

    if not args.load_weights and config_manager.check_workspace():
        exported_weight_path = config_manager.workspace_root / "models-exported/openvino.xml"
        if not exported_weight_path.exists():
            raise RuntimeError("OpenVINO-exported models are supported.")
        args.load_weights = str(exported_weight_path)

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

    if "save_model_to" not in args or not args.save_model_to:
        args.save_model_to = str(config_manager.workspace_root / "model-deployed")
    os.makedirs(args.save_model_to, exist_ok=True)
    task.deploy(deployed_model)
    with open(Path(args.save_model_to) / "openvino.zip", "wb") as write_file:
        write_file.write(deployed_model.exportable_code)

    return dict(retcode=0, template=template.name)


if __name__ == "__main__":
    main()
