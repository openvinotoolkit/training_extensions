"""Model optimization tool."""

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
import json
import os

from otx.api.configuration.helper import create
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import ModelEntity
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from otx.cli.datasets import get_dataset_class
from otx.cli.registry import find_and_parse_model_template
from otx.cli.utils.config import configure_dataset, override_parameters
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import generate_label_schema, read_model, save_model_data
from otx.cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    gen_params_dict_from_args,
)

# pylint: disable=too-many-locals


def parse_args():
    """Parses command line arguments.

    It dynamically generates help for hyper-parameters which are specific to particular model template.
    """

    pre_parser = argparse.ArgumentParser(add_help=False)
    if os.path.exists("./template.yaml"):
        template_path = "./template.yaml"
    else:
        pre_parser.add_argument("template")
        parsed, _ = pre_parser.parse_known_args()
        template_path = parsed.template
    # Load template.yaml file.
    template = find_and_parse_model_template(template_path)
    # Get hyper parameters schema.
    hyper_parameters = template.hyper_parameters.data
    assert hyper_parameters

    parser = argparse.ArgumentParser()
    if not os.path.exists("./template.yaml"):
        parser.add_argument("template")
    parser.add_argument("--data", required=False, default="./data.yaml")
    required = not os.path.exists("./data.yaml")

    parser.add_argument(
        "--train-ann-files",
        required=required,
        help="Comma-separated paths to training annotation files.",
    )
    parser.add_argument(
        "--train-data-roots",
        required=required,
        help="Comma-separated paths to training data folders.",
    )
    parser.add_argument(
        "--val-ann-files",
        required=required,
        help="Comma-separated paths to validation annotation files.",
    )
    parser.add_argument(
        "--val-data-roots",
        required=required,
        help="Comma-separated paths to validation data folders.",
    )
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load weights of trained model",
    )
    parser.add_argument(
        "--save-model-to",
        required=True,
        help="Location where trained model will be stored.",
    )
    parser.add_argument(
        "--save-performance",
        help="Path to a json file where computed performance will be stored.",
    )

    add_hyper_parameters_sub_parser(parser, hyper_parameters)

    return parser.parse_args(), template, hyper_parameters


def main():
    """Main function that is used for model training."""

    # Dynamically create an argument parser based on override parameters.
    args, template, hyper_parameters = parse_args()

    is_pot = False
    if args.load_weights.endswith(".bin") or args.load_weights.endswith(".xml"):
        is_pot = True

    if not is_pot and template.entrypoints.nncf is None:
        raise RuntimeError(f"Optimization by NNCF is not available for template {args.template}")

    # Get new values from user's input.
    updated_hyper_parameters = gen_params_dict_from_args(args)
    # Override overridden parameters by user's values.
    override_parameters(updated_hyper_parameters, hyper_parameters)

    hyper_parameters = create(hyper_parameters)

    # Get classes for Task, ConfigurableParameters and Dataset.
    task_class = get_impl_class(template.entrypoints.openvino if is_pot else template.entrypoints.nncf)
    dataset_class = get_dataset_class(template.task_type)

    data_config = configure_dataset(args)

    dataset = dataset_class(
        train_subset={
            "ann_file": data_config["data"]["train"]["ann-files"],
            "data_root": data_config["data"]["train"]["data-roots"],
        },
        val_subset={
            "ann_file": data_config["data"]["val"]["ann-files"],
            "data_root": data_config["data"]["val"]["data-roots"],
        },
    )

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=generate_label_schema(dataset, template.task_type),
        model_template=template,
    )

    environment.model = read_model(environment.get_model_configuration(), args.load_weights, None)

    task = task_class(task_environment=environment)

    output_model = ModelEntity(dataset, environment.get_model_configuration())

    task.optimize(
        OptimizationType.POT if is_pot else OptimizationType.NNCF,
        dataset,
        output_model,
        OptimizationParameters(),
    )

    save_model_data(output_model, args.save_model_to)

    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True),
    )

    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    task.evaluate(resultset)
    assert resultset.performance is not None
    print(resultset.performance)

    if args.save_performance:
        with open(args.save_performance, "w", encoding="UTF-8") as write_file:
            json.dump(
                {resultset.performance.score.name: resultset.performance.score.value},
                write_file,
            )


if __name__ == "__main__":
    main()
