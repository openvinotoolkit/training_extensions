"""Model quality evaluation tool."""

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

import json
from pathlib import Path

from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model_template import TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.cli.manager import ConfigManager
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import read_model
from otx.cli.utils.nncf import is_checkpoint_nncf
from otx.cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    get_parser_and_hprams_data,
)
from otx.core.data.adapter import get_dataset_adapter

# pylint: disable=too-many-locals


def get_args():
    """Parses command line arguments."""
    parser, hyper_parameters, params = get_parser_and_hprams_data()

    parser.add_argument(
        "--test-data-roots",
        help="Comma-separated paths to test data folders.",
    )
    parser.add_argument(
        "--load-weights",
        help="Load model weights from previously saved checkpoint."
        "It could be a trained/optimized model (POT only) or exported model.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Location where the intermediate output of the task will be stored.",
    )
    parser.add_argument(
        "--workspace",
        help="Path to the workspace where the command will run.",
        default=None,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="The data.yaml path want to use in train task.",
    )

    add_hyper_parameters_sub_parser(parser, hyper_parameters, modes=("INFERENCE",))
    override_param = [f"params.{param[2:].split('=')[0]}" for param in params if param.startswith("--")]

    return parser.parse_args(), override_param


def check_label_schemas(label_schema_a, label_schema_b):
    """Checks that both passed label schemas have labels with the same names.

    If it is False that it raises RuntimeError.
    """

    for model_label, snapshot_label in zip(label_schema_a.get_labels(False), label_schema_b.get_labels(False)):
        if model_label.name != snapshot_label.name:
            raise RuntimeError(
                "Labels schemas from model and dataset are different: " f"\n{label_schema_a} \n\tvs\n{label_schema_b}"
            )


def main():
    """Main function that is used for model evaluation."""

    # Dynamically create an argument parser based on override parameters.
    args, override_param = get_args()

    config_manager = ConfigManager(args, workspace_root=args.workspace, mode="eval")
    # Auto-Configuration for model template
    config_manager.configure_template()

    if not args.load_weights and config_manager.check_workspace():
        latest_model_path = (
            config_manager.workspace_root / "outputs" / "latest_trained_model" / "models" / "weights.pth"
        )
        args.load_weights = str(latest_model_path)

    # Update Hyper Parameter Configs
    hyper_parameters = config_manager.get_hyparams_config(override_param)

    # Get classes for Task, ConfigurableParameters and Dataset.
    template = config_manager.template
    if any(args.load_weights.endswith(x) for x in (".bin", ".xml", ".zip")):
        task_class = get_impl_class(template.entrypoints.openvino)
    elif args.load_weights.endswith(".pth"):
        if is_checkpoint_nncf(args.load_weights):
            task_class = get_impl_class(template.entrypoints.nncf)
        else:
            task_class = get_impl_class(template.entrypoints.base)
    else:
        raise ValueError(f"Unsupported file: {args.load_weights}")

    # Auto-Configuration for Dataset configuration
    config_manager.configure_data_config(update_data_yaml=config_manager.check_workspace())
    dataset_config = config_manager.get_dataset_config(subsets=["test"])
    dataset_adapter = get_dataset_adapter(**dataset_config)
    dataset, label_schema = dataset_adapter.get_otx_dataset(), dataset_adapter.get_label_schema()

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=label_schema,
        model_template=template,
    )

    environment.model = read_model(environment.get_model_configuration(), args.load_weights, None)

    task = task_class(task_environment=environment)

    validation_dataset = dataset.get_subset(Subset.TESTING)
    predicted_validation_dataset = task.infer(
        # temp (sungchul): remain annotation for visual prompting
        validation_dataset
        if getattr(task, "task_type", None) == TaskType.VISUAL_PROMPTING
        else validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=False),
    )

    resultset = ResultSetEntity(
        model=environment.model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    task.evaluate(resultset)
    assert resultset.performance is not None
    print(resultset.performance)

    output_path = Path(args.output) if args.output else config_manager.output_path
    with open(output_path / "performance.json", "w", encoding="UTF-8") as write_file:
        json.dump(
            {resultset.performance.score.name: resultset.performance.score.value},
            write_file,
        )

    return dict(retcode=0, template=template.name)


if __name__ == "__main__":
    main()
