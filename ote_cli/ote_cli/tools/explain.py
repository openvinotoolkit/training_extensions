"""
Model inference demonstration tool.
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
import time
from collections import deque

import cv2
import numpy as np
from ote_sdk.configuration.helper import create
from ote_sdk.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.datasets import DatasetEntity, DatasetItemEntity
from ote_sdk.entities.image import Image
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.task_environment import TaskEnvironment

from ote_cli.registry import find_and_parse_model_template
from ote_cli.tools.utils.demo.images_capture import open_images_capture
from ote_cli.utils.config import override_parameters
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.io import get_image_files, read_label_schema, read_model, save_saliency_output
from ote_cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    gen_params_dict_from_args,
)

ESC_BUTTON = 27
SUPPORTED_EXPLAIN_ALGORITHMS = ["CAM", "EigenCAM"]


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
        "-i",
        "--input",
        required=True,
        help="Source of input data: images folder, image, webcam and video.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="saliency_dump",
        help="Output path for explanation images.",
    )
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load only weights from previously saved checkpoint",
    )
    parser.add_argument(
        "--explain-model",
        default="EigenCAM",
        help=f"XAI model name, currently support {SUPPORTED_EXPLAIN_ALGORITHMS}",
    )
    add_hyper_parameters_sub_parser(parser, hyper_parameters, modes=("INFERENCE",))

    return parser.parse_args(), template, hyper_parameters


def get_explain_map(task, frame):
    """
    Returns list of explanations made by task on frame and time spent on doing explanation.
    """

    empty_annotation = AnnotationSceneEntity(
        annotations=[], kind=AnnotationSceneKind.PREDICTION
    )

    item = DatasetItemEntity(
        media=Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        annotation_scene=empty_annotation,
    )

    dataset = DatasetEntity(items=[item])

    start_time = time.perf_counter()
    explain_map = task.explain(
        dataset,
        InferenceParameters(is_evaluation=True),
    )
    elapsed_time = time.perf_counter() - start_time
    saliency_map = explain_map[0]
    return saliency_map, elapsed_time


def main():
    """
    Main function that is used for model demonstration.
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
        task_class = get_impl_class(template.entrypoints.base)
    else:
        raise ValueError(f"Unsupported file: {args.load_weights}.")

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

    imgs = get_image_files(args.input)
    if imgs is None:
        raise ValueError(f"No image found in {args.input}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    elapsed_times = deque(maxlen=10)

    for _, (root_dir, filename) in enumerate(imgs):
        img = cv2.imread(os.path.join(root_dir, filename))
        saliency_map, elapsed_time = get_explain_map(task, img)
        elapsed_times.append(elapsed_time)
        save_saliency_output(img, saliency_map.numpy, root_dir, filename.split[0])

    print(f"saliency maps saved to {args.output}...")


if __name__ == "__main__":
    main()
