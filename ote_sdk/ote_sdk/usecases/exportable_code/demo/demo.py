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
from ote_sdk.usecases.exportable_code.demo.demo_package import (
    AsyncInferencer,
    ChainInferencer,
    SyncInferencer,
    create_model,
    create_output_converter,
    create_visualizer,
)


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
        "--models",
        help="Required. Path to an .xml files with a trained models.",
        nargs="+",
        required=True,
        type=Path,
    )
    args.add_argument(
        "-it",
        "--inference_type",
        help="Optional. Type of inference. For task-chain you should type 'chain'.",
        choices=["sync", "async", "chain"],
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


INFERENCER = {
    "sync": SyncInferencer,
    "async": AsyncInferencer,
    "chain": ChainInferencer,
}


def get_inferencer_class(type_inference, models):
    """
    Return class for inference of models
    """
    if type_inference == "chain" and len(models) == 1:
        raise RuntimeError(
            "For single model please use 'sync' or 'async' type of inference"
        )
    if len(models) > 1 and type_inference != "chain":
        raise RuntimeError(
            "For task-chain scenario please use 'chain' type of inference"
        )
    return INFERENCER[type_inference]


def main():
    """
    Main function that is used to run demo.
    """
    args = build_argparser().parse_args()
    # create models and converters for outputs
    models = []
    converters = []
    last_config = ""
    for model_path in args.models:
        config_file = model_path.parent.resolve() / "config.json"
        last_config = config_file
        model_file = model_path.parent.parent.resolve() / "python" / "model.py"
        model_file = model_file if model_file.exists() else None
        models.append(create_model(model_path, config_file, model_file))
        converters.append(create_output_converter(config_file))

    # create visualizer
    visualizer = create_visualizer(last_config, args.inference_type)

    # create inferencer and run
    demo = get_inferencer_class(args.inference_type, models)(
        models, converters, visualizer
    )
    demo.run(args.input, args.loop)


if __name__ == "__main__":
    sys.exit(main() or 0)
