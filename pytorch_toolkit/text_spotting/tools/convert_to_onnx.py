"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import json
import logging

from segmentoly.utils.logging import setup_logging
from segmentoly.utils.stats import add_flops_counting_methods, flops_to_string, \
    print_model_with_flops
from segmentoly.utils.weights import load_checkpoint
from text_spotting.models.text_detectors import make_text_detector
from text_spotting.utils.onnx import onnx_export, export_to_onnx_text_recognition_encoder, \
    export_to_onnx_text_recognition_decoder


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser(description='Export Mask R-CNN network to ONNX')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to configuration file implementing text spotting model.')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Checkpoint file path to load network weights from.')
    parser.add_argument('--input_size', type=int, nargs=2, required=True,
                        help='Input resolution.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the output ONNX file.')
    parser.add_argument('--verbose', action='store_true',
                        help='Run export in verbose mode.')
    parser.add_argument('--check', action='store_true',
                        help='Run ONNX model checker after export.')
    parser.add_argument('--show_flops', action='store_true',
                        help='Estimate computational complexity of the network.')

    return parser.parse_args()


def main():
    """ Does export to onnx. """

    args = parse_args()

    with open(args.model) as file:
        config = json.load(file)
    text_spotter = make_text_detector(**config['model'])(2, fc_detection_head=False,
                                                         shape=args.input_size)
    load_checkpoint(text_spotter, args.ckpt)
    text_spotter.export_mode = True

    net = text_spotter

    if args.show_flops:
        net = add_flops_counting_methods(net)
        net.reset_flops_count()
        net.start_flops_count()

    # Export of text detection part (Mask-RCNN subgraph).
    printable_graph = onnx_export(net, args.input_size, args.output_folder, check=args.check,
                                  verbose=args.verbose)
    if args.verbose:
        logging.info(printable_graph)

    if args.show_flops:
        net.stop_flops_count()
        logging.info('Computational complexity of text detection part: {}'.format(
            flops_to_string(net.compute_average_flops_cost())))
        if args.verbose:
            print_model_with_flops(net)

    # Export of text recognition encoder
    net = text_spotter.text_recogn_head.encoder
    if args.show_flops:
        net = add_flops_counting_methods(net)
        net.reset_flops_count()
        net.start_flops_count()

    printable_graph = export_to_onnx_text_recognition_encoder(
        net, text_spotter.text_recogn_head.input_feature_size, args.output_folder)
    if args.verbose:
        logging.info(printable_graph)

    if args.show_flops:
        net.stop_flops_count()
        logging.info('Computational complexity of text recognition encoder part: {}'.format(
            flops_to_string(net.compute_average_flops_cost())))
        if args.verbose:
            print_model_with_flops(net)

    # Export of text recognition decoder
    net = text_spotter.text_recogn_head.decoder
    if args.show_flops:
        net = add_flops_counting_methods(net)
        net.reset_flops_count()
        net.start_flops_count()

    printable_graph = export_to_onnx_text_recognition_decoder(
        net, text_spotter.text_recogn_head.input_feature_size, args.output_folder)
    if args.verbose:
        logging.info(printable_graph)

    if args.show_flops:
        net.stop_flops_count()
        logging.info('Computational complexity of text recognition decoder part: {}'.format(
            flops_to_string(net.compute_average_flops_cost())))
        if args.verbose:
            print_model_with_flops(net)


if __name__ == '__main__':
    setup_logging()
    main()
