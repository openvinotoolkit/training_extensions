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

import argparse
import json
import os

from otx.api.configuration.helper import create
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.cli.registry import find_and_parse_model_template
from otx.cli.utils.config import configure_dataset, override_parameters
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import read_model
from otx.cli.utils.nncf import is_checkpoint_nncf
from otx.cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    gen_params_dict_from_args,
)
from otx.core.data.adapter import get_dataset_adapter

# pylint: disable=too-many-locals


def parse_args():
    """Parses command line arguments."""

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
        "--test-data-roots",
        required=required,
        help="Comma-separated paths to test data folders.",
    )
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load model weights from previously saved checkpoint."
        "It could be a trained/optimized model (POT only) or exported model.",
    )
    parser.add_argument(
        "--save-performance",
        help="Path to a json file where computed performance will be stored.",
    )

    add_hyper_parameters_sub_parser(parser, hyper_parameters, modes=("INFERENCE",))

    return parser.parse_args(), template, hyper_parameters


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
    args, template, hyper_parameters = parse_args()
    # Get new values from user's input.
    updated_hyper_parameters = gen_params_dict_from_args(args)
    # Override overridden parameters by user's values.
    override_parameters(updated_hyper_parameters, hyper_parameters)

    hyper_parameters = create(hyper_parameters)

    # Get classes for Task, ConfigurableParameters and Dataset.
    if any(args.load_weights.endswith(x) for x in (".bin", ".xml", ".zip")):
        task_class = get_impl_class(template.entrypoints.openvino)
    elif args.load_weights.endswith(".pth"):
        if is_checkpoint_nncf(args.load_weights):
            task_class = get_impl_class(template.entrypoints.nncf)
        else:
            task_class = get_impl_class(template.entrypoints.base)
    else:
        raise ValueError(f"Unsupported file: {args.load_weights}")

    data_config = configure_dataset(args)

    data_roots = dict(
        test_subset={
            "data_root": data_config["data"]["test"]["data-roots"],
        }
    )

    dataset_adapter = get_dataset_adapter(
        template.task_type,
        test_data_roots=data_roots["test_subset"]["data_root"]
    )

    dataset = dataset_adapter.get_otx_dataset()
    label_schema = dataset_adapter.get_label_schema()

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
        validation_dataset.with_empty_annotations(),
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

    if not args.save_performance:
        args.save_performance = os.path.join(os.path.dirname(args.load_weights), "performance.json")
    with open(args.save_performance, "w", encoding="UTF-8") as write_file:
        json.dump(
            {resultset.performance.score.name: resultset.performance.score.value},
            write_file,
        )


if __name__ == "__main__":
    main()
