"""
Model quality evaluation tool.
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
import os

from ote_sdk.configuration.helper import create
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment

from ote_cli.datasets import get_dataset_class
from ote_cli.registry import find_and_parse_model_template
from ote_cli.utils.config import override_parameters
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.io import generate_label_schema, read_label_schema, read_model
from ote_cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    gen_params_dict_from_args,
)


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
        "--test-ann-files",
        required=True,
        help="Comma-separated paths to test annotation files.",
    )
    parser.add_argument(
        "--test-data-roots",
        required=True,
        help="Comma-separated paths to test data folders.",
    )
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load only weights from previously saved checkpoint",
    )
    parser.add_argument(
        "--save-performance",
        help="Path to a json file where computed performance will be stored.",
    )

    add_hyper_parameters_sub_parser(parser, hyper_parameters, modes=("INFERENCE",))

    return parser.parse_args(), template, hyper_parameters


def check_label_schemas(label_schema_a, label_schema_b):
    """
    Checks that both passed label schemas have labels with the same names.
    If it is False that it raises RuntimeError.
    """

    for model_label, snapshot_label in zip(
        label_schema_a.get_labels(False), label_schema_b.get_labels(False)
    ):
        if model_label.name != snapshot_label.name:
            raise RuntimeError(
                "Labels schemas from model and dataset are different: "
                f"\n{label_schema_a} \n\tvs\n{label_schema_b}"
            )


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
    if args.load_weights.endswith(".bin") or args.load_weights.endswith(".xml"):
        task_class = get_impl_class(template.entrypoints.openvino)
    else:
        task_class = get_impl_class(template.entrypoints.base)

    dataset_class = get_dataset_class(template.task_type)

    dataset = dataset_class(
        test_subset={"ann_file": args.test_ann_files, "data_root": args.test_data_roots}
    )

    dataset_label_schema = generate_label_schema(dataset, template.task_type)
    check_label_schemas(
        read_label_schema(
            os.path.join(os.path.dirname(args.load_weights), "label_schema.json")
        ),
        dataset_label_schema,
    )

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=dataset_label_schema,
        model_template=template,
    )

    model = read_model(environment.get_model_configuration(), args.load_weights, None)
    environment.model = model

    task = task_class(task_environment=environment)

    validation_dataset = dataset.get_subset(Subset.TESTING)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True),
    )

    resultset = ResultSetEntity(
        model=model,
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
