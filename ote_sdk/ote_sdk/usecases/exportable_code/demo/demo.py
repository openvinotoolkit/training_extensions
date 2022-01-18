"""
Demo based on ModelAPI
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import sys
from argparse import SUPPRESS, ArgumentParser
from pathlib import Path

# pylint: disable=no-name-in-module, import-error
from demo_package import SyncDemo, create_model, create_output_converter

from ote_sdk.usecases.exportable_code.streamer import get_media_type
from ote_sdk.usecases.exportable_code.visualization import Visualizer


def build_argparser():
    """
    Parses command line arguments.
    """
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
        help="Required. Path to an .xml file with a trained model.",
        required=True,
        type=Path,
    )
    args.add_argument(
        "-c",
        "--config",
        help="Optional. Path to an .json file with parameters for model.",
        type=Path,
    )

    return parser


def main():
    """
    Main function that is used to run demo.
    """
    args = build_argparser().parse_args()
    # create components for demo

    model = create_model(args.model, args.config)
    media_type = get_media_type(args.input)

    visualizer = Visualizer(media_type)
    converter = create_output_converter(args.config)
    demo = SyncDemo(model, visualizer, converter)
    demo.run(args.input)


if __name__ == "__main__":
    sys.exit(main() or 0)
