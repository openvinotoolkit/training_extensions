"""
Model optimization tool.
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
import json

from ote_sdk.configuration.helper import create
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType

from ote_cli.datasets import get_dataset_class
from ote_cli.registry import find_and_parse_model_template
from ote_cli.utils.config import override_parameters
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.io import generate_label_schema, read_model, save_model_data
from ote_cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    gen_params_dict_from_args,
)


def parse_args():
    """
    Parses command line arguments.
    It dynamically generates help for hyper-parameters which are specific to particular model template.
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
        "--train-ann-files",
        required=True,
        help="Comma-separated paths to training annotation files.",
    )
    parser.add_argument(
        "--train-data-roots",
        required=True,
        help="Comma-separated paths to training data folders.",
    )
    parser.add_argument(
        "--val-ann-files",
        required=True,
        help="Comma-separated paths to validation annotation files.",
    )
    parser.add_argument(
        "--val-data-roots",
        required=True,
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
    """
    Main function that is used for model training.
    """

    # Dynamically create an argument parser based on override parameters.
    args, template, hyper_parameters = parse_args()

    is_pot = False
    if args.load_weights.endswith(".bin") or args.load_weights.endswith(".xml"):
        is_pot = True

    if not is_pot and template.entrypoints.nncf is None:
        raise RuntimeError(
            f"Optimization by NNCF is not available for template {args.template}"
        )

    # Get new values from user's input.
    updated_hyper_parameters = gen_params_dict_from_args(args)
    # Override overridden parameters by user's values.
    override_parameters(updated_hyper_parameters, hyper_parameters)

    hyper_parameters = create(hyper_parameters)

    # Get classes for Task, ConfigurableParameters and Dataset.
    task_class = get_impl_class(
        template.entrypoints.openvino if is_pot else template.entrypoints.nncf
    )
    dataset_class = get_dataset_class(template.task_type)

    # Create instances of Task, ConfigurableParameters and Dataset.
    dataset = dataset_class(
        train_subset={
            "ann_file": args.train_ann_files,
            "data_root": args.train_data_roots,
        },
        val_subset={"ann_file": args.val_ann_files, "data_root": args.val_data_roots},
    )

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=generate_label_schema(dataset, template.task_type),
        model_template=template,
    )

    environment.model = read_model(
        environment.get_model_configuration(), args.load_weights, None
    )

    task = task_class(task_environment=environment)

    output_model = ModelEntity(dataset, environment.get_model_configuration())

    task.optimize(
        OptimizationType.POT if is_pot else OptimizationType.NNCF,
        dataset,
        output_model,
        None,
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
