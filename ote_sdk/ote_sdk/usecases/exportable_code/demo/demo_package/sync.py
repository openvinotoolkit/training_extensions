"""
Sync Demo based on ModelAPI
"""
# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

import sys
from argparse import SUPPRESS, ArgumentParser, Namespace
from pathlib import Path

from ote_sdk.usecases.exportable_code.demo.demo_package.utils import (
    create_model,
    create_output_converter,
)
from ote_sdk.usecases.exportable_code.streamer import get_media_type, get_streamer
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
        help="Required. An input to process. The input must be a single image.",
    )
    args.add_argument(
        "-m",
        "--model",
        help="Required. Path to an .xml file with a trained model.",
        required=True,
        type=Path,
    )

    return parser


class SyncDemo:
    """
    Synd demo for model inference

    Args:
        model: model for inference
        visualizer: for visualize inference results
        converter: convert model ourtput to annotation scene
    """

    def __init__(self, model, visualizer, converter) -> None:
        self.model = model
        self.visualizer = visualizer
        self.converter = converter

    def run(self, input_stream):
        """
        Run demo using input stream (image, video stream, camera)
        """
        self.streamer = get_streamer(input_stream)
        for frame in self.streamer:
            # getting result include preprocessing, infer, postprocessing for sync infer
            dict_data, input_meta = self.model.preprocess(frame)
            raw_result = self.model.infer_sync(dict_data)
            predictions = self.model.postprocess(raw_result, input_meta)
            annotation_scene = self.converter.convert_to_annotation(
                predictions, input_meta
            )

            # any user's visualizer
            output = self.visualizer.draw(frame, annotation_scene)
            self.visualizer.show(output)

            if self.visualizer.is_quit():
                break


def main():
    """
    Main function that is used for run demo.
    """
    args = build_argparser().parse_args()
    # create components for demo

    model = create_model(args.model)
    media_type = get_media_type(args.input)

    visualizer = Visualizer(media_type)
    converter = create_output_converter(model.labels)
    demo = SyncDemo(model, visualizer, converter)
    demo.run(args.input)


if __name__ == "__main__":
    sys.exit(main() or 0)
