"""
Model explain demonstration tool.
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
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.task_environment import TaskEnvironment

from ote_cli.registry import find_and_parse_model_template
from ote_cli.utils.config import override_parameters
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.io import (
    get_explain_dataset_from_filelist,
    get_image_files,
    read_label_schema,
    read_model,
    save_saliency_output,
)
from ote_cli.utils.nncf import is_checkpoint_nncf
from ote_cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    gen_params_dict_from_args,
)

ESC_BUTTON = 27
SUPPORTED_EXPLAIN_ALGORITHMS = ["ActivationMap", "EigenCAM"]


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
        "--explain-data-roots",
        required=True,
        help="Comma-separated paths to explain data folders.",
    )
    parser.add_argument(
        "--save-explanation-to",
        default="saliency_dump",
        help="Output path for explanation images.",
    )
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load only weights from previously saved checkpoint",
    )
    parser.add_argument(
        "--explain-algorithm",
        default="EigenCAM",
        help=f"Explain algorithm name, currently support {SUPPORTED_EXPLAIN_ALGORITHMS}",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=float,
        default=0.5,
        help="weight of the saliency map when overlaying the saliency map",
    )
    add_hyper_parameters_sub_parser(parser, hyper_parameters, modes=("INFERENCE",))

    return parser.parse_args(), template, hyper_parameters


def main():
    """
    Main function that is used for model explanation.
    """

    # Dynamically create an argument parser based on override parameters.
    args, template, hyper_parameters = parse_args()
    # Get new values from user's input.
    updated_hyper_parameters = gen_params_dict_from_args(args)
    # Override overridden parameters by user's values.
    override_parameters(updated_hyper_parameters, hyper_parameters)

    hyper_parameters = create(hyper_parameters)

    # Get classes for Task, ConfigurableParameters and Dataset.
    if args.load_weights.endswith(".pth"):
        if is_checkpoint_nncf(args.load_weights):
            task_class = get_impl_class(template.entrypoints.nncf)
        else:
            task_class = get_impl_class(template.entrypoints.base)
    else:
        raise ValueError(f"Unsupported file: {args.load_weights}")

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=read_label_schema(args.load_weights),
        model_template=template,
    )

    environment.model = read_model(
        environment.get_model_configuration(), args.load_weights, None
    )

    task = task_class(task_environment=environment)

    if args.explain_algorithm not in SUPPORTED_EXPLAIN_ALGORITHMS:
        raise NotImplementedError(
            f"{args.explain_algorithm} currently not supported. \
            Currently only support {SUPPORTED_EXPLAIN_ALGORITHMS}"
        )
    if not os.path.exists(args.save_explanation_to):
        os.makedirs(args.save_explanation_to)

    image_files = get_image_files(args.explain_data_roots)
    dataset_to_explain = get_explain_dataset_from_filelist(image_files)
    explained_dataset = task.explain(
        dataset_to_explain.with_empty_annotations(),
        InferenceParameters(
            is_evaluation=True,
            explainer=args.explain_algorithm,
        ),
    )

    for explained_data, (_, fname) in zip(explained_dataset, image_files):
        save_saliency_output(
            explained_data.numpy,
            explained_data.get_metadata()[0].data.numpy,
            args.save_explanation_to,
            os.path.splitext(os.path.basename(fname))[0],
            weight=args.weight,
        )

    print(
        f"saliency maps saved to {args.save_explanation_to} for {len(image_files)} images..."
    )


if __name__ == "__main__":
    main()
