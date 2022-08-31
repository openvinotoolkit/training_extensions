"""Demo based on ModelAPI."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import sys
from argparse import SUPPRESS, ArgumentParser
from pathlib import Path

# pylint: disable=no-name-in-module, import-error
from otx.api.usecases.exportable_code.demo.demo_package import (
    AsyncExecutor,
    ChainExecutor,
    ModelContainer,
    SyncExecutor,
    create_visualizer,
)


def build_argparser():
    """Parses command line arguments."""
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument(
        "-h",
        "--help",
        action="help",
        default=SUPPRESS,
        help="Show this help message and exit.",
    )
    args.add_argument(
        "-i",
        "--input",
        required=True,
        help="Required. An input to process. The input must be a single image, "
        "a folder of images, video file or camera id.",
    )
    args.add_argument(
        "-m",
        "--models",
        help="Required. Path to directory with trained model and configuration file. "
        "If you provide several models you will start the task chain pipeline with "
        "the provided models in the order in which they were specified.",
        nargs="+",
        required=True,
        type=Path,
    )
    args.add_argument(
        "-it",
        "--inference_type",
        help="Optional. Type of inference for single model.",
        choices=["sync", "async"],
        default="sync",
        type=str,
    )
    args.add_argument(
        "-l",
        "--loop",
        help="Optional. Enable reading the input in a loop.",
        default=False,
        action="store_true",
    )

    return parser


EXECUTORS = {
    "sync": SyncExecutor,
    "async": AsyncExecutor,
    "chain": ChainExecutor,
}


def get_inferencer_class(type_inference, models):
    """Return class for inference of models."""
    if len(models) > 1:
        type_inference = "chain"
        print("You started the task chain pipeline with the provided models in the order in which they were specified")
    return EXECUTORS[type_inference]


def main():
    """Main function that is used to run demo."""
    args = build_argparser().parse_args()
    # create models
    models = []
    for model_dir in args.models:
        model = ModelContainer(model_dir)
        models.append(model)

    inferencer = get_inferencer_class(args.inference_type, models)

    # create visualizer
    visualizer = create_visualizer(models[-1].task_type)

    if len(models) == 1:
        models = models[0]

    # create inferencer and run
    demo = inferencer(models, visualizer)
    demo.run(args.input, args.loop)


if __name__ == "__main__":
    sys.exit(main() or 0)
