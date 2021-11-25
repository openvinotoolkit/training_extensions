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
from argparse import ArgumentParser, Namespace, SUPPRESS
from pathlib import Path


from ote_sdk.usecases.exportable_code.streamer import get_media_type, get_streamer
from ote_sdk.usecases.exportable_code.visualization import Visualizer

from .utils import create_model, create_output_converter


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      default=None, type=Path)
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    return parser


def get_infer_parameters(args):
    params = {
        'device': args.device,
        'streams': args.num_streams,
        'threads': args.num_threads,
        'infer_requests': args.num_infer_requests
    }
    return params


class SyncDemo:
    def __init__(self, model, visualizer, converter) -> None:
        self.model = model
        self.visualizer = visualizer
        self.converter = converter

    def run(self, input_stream):
        self.streamer = get_streamer(input_stream)
        for frame in self.streamer:
            # getting result include preprocessing, infer, postprocessing for sync infer
            dict_data, input_meta = self.model.preprocess(frame)
            raw_result = self.model.infer_sync(dict_data)
            predictions = self.model.postprocess(raw_result, input_meta)
            annotation_scene = self.converter.convert_to_annotation(predictions, input_meta)

            # any user's visualizer
            output = self.visualizer.draw(frame, annotation_scene)
            self.visualizer.show(output)

            if self.visualizer.is_quit():
                break


def main():
    args = build_argparser().parse_args()
    # create components for demo

    model = create_model(Namespace(**get_infer_parameters(args)), args.model)
    media_type = get_media_type(args.input)

    visualizer = Visualizer(media_type)
    converter = create_output_converter(model.labels)
    demo = SyncDemo(model, visualizer, converter)
    demo.run(args.input)


if __name__ == '__main__':
    sys.exit(main() or 0)
