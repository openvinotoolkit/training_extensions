"""Model exporting tool."""

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

from pathlib import Path

from otx.api.configuration.helper import create
from otx.api.entities.model import ModelEntity, ModelOptimizationType, ModelPrecision
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.cli.manager import ConfigManager
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import read_binary, read_label_schema, save_model_data
from otx.cli.utils.nncf import is_checkpoint_nncf
from otx.cli.utils.parser import get_parser_and_hprams_data


def get_args():
    """Parses command line arguments."""
    parser, _, _ = get_parser_and_hprams_data()

    parser.add_argument(
        "--load-weights",
        help="Load model weights from previously saved checkpoint.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Location where exported model will be stored.",
    )
    parser.add_argument(
        "--workspace",
        help="Path to the workspace where the command will run.",
        default=None,
    )
    parser.add_argument(
        "--dump-features",
        action="store_true",
        help="Whether to return feature vector and saliency map for explanation purposes.",
    )
    parser.add_argument(
        "--half-precision",
        action="store_true",
        help="This flag indicated if model is exported in half precision (FP16).",
    )
    parser.add_argument(
        "--export-type",
        help="Type of the resulting model (OpenVINO or ONNX).",
        default="openvino",
    )

    return parser.parse_args()


def main():
    """Main function that is used for model exporting."""
    args = get_args()
    config_manager = ConfigManager(args, mode="export", workspace_root=args.workspace)
    # Auto-Configuration for model template
    config_manager.configure_template()

    # Load template.yaml file.
    template = config_manager.template

    # Get class for Task.
    if not args.load_weights and config_manager.check_workspace():
        latest_model_path = (
            config_manager.workspace_root / "outputs" / "latest_trained_model" / "models" / "weights.pth"
        )
        args.load_weights = str(latest_model_path)

    is_nncf = is_checkpoint_nncf(args.load_weights)
    task_class = get_impl_class(template.entrypoints.nncf if is_nncf else template.entrypoints.base)

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
        optimization_type=ModelOptimizationType.NNCF if is_nncf else ModelOptimizationType.NONE,
    )
    environment.model = model

    (config_manager.output_path / "logs").mkdir(exist_ok=True, parents=True)
    task = task_class(task_environment=environment, output_path=str(config_manager.output_path / "logs"))

    exported_model = ModelEntity(None, environment.get_model_configuration())

    export_precision = ModelPrecision.FP16 if args.half_precision else ModelPrecision.FP32

    if args.export_type.lower() not in ["openvino", "onnx"]:
        raise ValueError("Unsupported export type")
    export_type = ExportType.OPENVINO if "openvino" == args.export_type.lower() else ExportType.ONNX
    task.export(export_type, exported_model, export_precision, args.dump_features)

    if not args.output:
        output_path = config_manager.output_path
        output_path = output_path / "openvino"
    else:
        output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    save_model_data(exported_model, str(output_path))

    return dict(retcode=0, template=template.name)


if __name__ == "__main__":
    main()
