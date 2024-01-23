"""Model inference demonstration tool."""

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

import time
from collections import deque

import cv2
import numpy as np

# Update environment variables for CLI use
import otx.cli  # noqa: F401
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.datasets import DatasetEntity, DatasetItemEntity
from otx.api.entities.image import Image
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.utils.vis_utils import dump_frames
from otx.cli.manager import ConfigManager
from otx.cli.tools.utils.demo.images_capture import open_images_capture
from otx.cli.tools.utils.demo.visualization import draw_predictions, put_text_on_rect_bg
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import read_label_schema, read_model
from otx.cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    get_override_param,
    get_parser_and_hprams_data,
)

ESC_BUTTON = 27


def get_args():
    """Parses command line arguments."""
    parser, hyper_parameters, params = get_parser_and_hprams_data()

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Source of input data: images folder, image, webcam and video.",
    )
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load model weights from previously saved checkpoint."
        "It could be a trained/optimized model (POT only) or exported model.",
    )
    parser.add_argument(
        "--fit-to-size",
        nargs=2,
        type=int,
        help="Width and Height space-separated values. "
        "Fits displayed images to window with specified Width and Height. "
        "This options applies to result visualisation only.",
    )
    parser.add_argument("--loop", action="store_true", help="Enable reading the input in a loop.")
    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Frame visualization time in ms. Negative delay value disables visualization",
    )
    parser.add_argument(
        "--display-perf",
        action="store_true",
        help="This option enables writing performance metrics on displayed frame. "
        "These metrics take into account not only model inference time, but also "
        "frame reading, pre-processing and post-processing.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Output path to save input data with predictions.",
    )

    add_hyper_parameters_sub_parser(parser, hyper_parameters, modes=("INFERENCE",))
    override_param = get_override_param(params)

    return parser.parse_args(), override_param


def get_predictions(task, frame):
    """Returns list of predictions made by task on frame and time spent on doing prediction."""

    empty_annotation = AnnotationSceneEntity(annotations=[], kind=AnnotationSceneKind.PREDICTION)

    item = DatasetItemEntity(
        media=Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        annotation_scene=empty_annotation,
    )

    dataset = DatasetEntity(items=[item])

    start_time = time.perf_counter()
    predicted_validation_dataset = task.infer(
        dataset,
        InferenceParameters(is_evaluation=False),
    )
    elapsed_time = time.perf_counter() - start_time
    item = predicted_validation_dataset[0]
    return item.get_annotations(), elapsed_time


def main():
    """Main function that is used for model demonstration."""

    # Dynamically create an argument parser based on override parameters.
    args, override_param = get_args()

    if args.loop and args.output:
        raise ValueError("--loop and --output cannot be both specified")

    config_manager = ConfigManager(args, mode="demo")
    # Auto-Configuration for model template
    config_manager.configure_template()

    # Update Hyper Parameter Configs
    hyper_parameters = config_manager.get_hyparams_config(override_param)

    # Get classes for Task, ConfigurableParameters and Dataset.
    template = config_manager.template
    if any(args.load_weights.endswith(x) for x in (".bin", ".xml", ".zip")):
        task_class = get_impl_class(template.entrypoints.openvino)
    elif args.load_weights.endswith(".pth"):
        task_class = get_impl_class(template.entrypoints.base)
    else:
        raise ValueError(f"Unsupported file: {args.load_weights}")

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=read_label_schema(args.load_weights),
        model_template=template,
    )

    environment.model = read_model(environment.get_model_configuration(), args.load_weights, None)

    task = task_class(task_environment=environment)

    capture = open_images_capture(args.input, args.loop)

    elapsed_times = deque(maxlen=10)
    saved_frames = []
    while True:
        frame = capture.read()
        if frame is None:
            break

        predictions, elapsed_time = get_predictions(task, frame)
        elapsed_times.append(elapsed_time)
        elapsed_time = np.mean(elapsed_times)

        frame = draw_predictions(template.task_type, predictions, frame, args.fit_to_size)
        if args.display_perf:
            put_text_on_rect_bg(
                frame,
                f"time: {elapsed_time:.4f} sec.",
                (0, frame.shape[0] - 30),
                color=(255, 255, 255),
            )

        if args.delay > 0:
            cv2.imshow("frame", frame)
            if cv2.waitKey(args.delay) == ESC_BUTTON:
                break
        else:
            print(f"Frame: {elapsed_time=}, {len(predictions)=}")

        if args.output:
            saved_frames.append(frame)

    dump_frames(saved_frames, args.output, args.input, capture)

    return dict(retcode=0, template=template.name)


if __name__ == "__main__":
    main()
