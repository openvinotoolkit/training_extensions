# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Demo based on ModelAPI."""

import sys
from argparse import SUPPRESS, ArgumentParser
from pathlib import Path

from demo_package import AsyncExecutor, ModelWrapper, SyncExecutor, create_visualizer


def build_argparser() -> ArgumentParser:
    """Returns an ArgumentParser for parsing command line arguments."""
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
        "--model",
        help="Optional. Path to directory with trained model and configuration file. "
        "Default value points to deployed model folder '../model'.",
        default=Path("../model"),
        type=Path,
    )
    args.add_argument(
        "-it",
        "--inference_type",
        help="Optional. Type of inference for single model.",
        choices=["sync", "async"],
        default="async",
        type=str,
    )
    args.add_argument(
        "-l",
        "--loop",
        help="Optional. Enable reading the input in a loop.",
        default=False,
        action="store_true",
    )
    args.add_argument(
        "--no_show",
        help="Optional. Disables showing inference results on UI.",
        default=False,
        action="store_true",
    )
    args.add_argument(
        "-d",
        "--device",
        help="Optional. Device to infer the model.",
        choices=["CPU", "GPU"],
        default="CPU",
        type=str,
    )
    args.add_argument(
        "--output",
        default="./outputs/model_visualization",
        type=str,
        help="Optional. Output path to save input data with predictions.",
    )

    return parser


EXECUTORS = {
    "sync": SyncExecutor,
    "async": AsyncExecutor,
}


def main() -> int:
    """Main function that is used to run demo."""
    args = build_argparser().parse_args()

    if args.loop and args.output:
        msg = "--loop and --output cannot be both specified"
        raise ValueError(msg)

    # create models
    model = ModelWrapper(args.model, device=args.device)
    inferencer = EXECUTORS[args.inference_type]

    # create visualizer
    visualizer = create_visualizer(model.task_type, model.labels, no_show=args.no_show, output=args.output)

    # create inferencer and run
    demo = inferencer(model, visualizer)
    demo.run(args.input, args.loop and not args.no_show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
