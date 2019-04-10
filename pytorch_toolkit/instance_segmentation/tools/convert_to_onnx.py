"""
 Copyright (c) 2019 Intel Corporation

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
import logging
from pydoc import locate

from segmentoly.datasets.factory import get_dataset
from segmentoly.utils.logging import setup_logging
from segmentoly.utils.onnx import onnx_export
from segmentoly.utils.stats import add_flops_counting_methods, flops_to_string, print_model_with_flops
from segmentoly.utils.weights import load_checkpoint


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Export Mask R-CNN network to ONNX')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to module and class implementing Mask R-CNN instance segmentation model.')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Checkpoint file path to load network weights from.')
    parser.add_argument('--input_size', type=int, nargs=2, required=True,
                        help='Input resolution.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output ONNX file.')

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--dataset', type=str, default=None,
                               help='Dataset name.')
    dataset_group.add_argument('-nc', '--classes_num', type=int, default=None,
                               help='Number of supported object classes.')

    parser.add_argument('--verbose', action='store_true',
                        help='Run export in verbose mode.')
    parser.add_argument('--check', action='store_true',
                        help='Run ONNX model checker after export.')
    parser.add_argument('--show_flops', action='store_true',
                        help='Estimate computational complexity of the network.')
    return parser.parse_args()


def main(args):

    if args.dataset:
        dataset = get_dataset(args.dataset, False, False, None)
        classes_num = dataset.classes_num
    else:
        classes_num = args.classes_num

    net = locate(args.model)(classes_num, fc_detection_head=False)
    load_checkpoint(net, args.ckpt)

    if args.show_flops:
        net = add_flops_counting_methods(net)
        net.reset_flops_count()
        net.start_flops_count()

    printable_graph = onnx_export(net, args.input_size, args.output_file, check=args.check, verbose=args.verbose)

    if args.verbose:
        logging.info(printable_graph)

    if args.show_flops:
        net.stop_flops_count()
        logging.info('Computational complexity: {}'.format(flops_to_string(net.compute_average_flops_cost())))
        if args.verbose:
            print_model_with_flops(net)


if __name__ == '__main__':
    setup_logging()
    args = parse_args()
    main(args)
