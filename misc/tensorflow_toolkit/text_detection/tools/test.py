#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
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

""" This module performs testing of text detection neural network. """

import argparse
import os
import tensorflow as tf

from text_detection.model import pixel_link_model
from text_detection.metrics import test
from text_detection.common import load_config, parse_epoch


def arg_parser():
    """ Returns argument parser. """

    parser = argparse.ArgumentParser(description='Runs an evaluation of text detection.')

    parser.add_argument('--weights')
    parser.add_argument('--weights_folder')
    parser.add_argument('--resolution', nargs=2, type=int, default=(1280, 768))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--imshow_delay', type=int, default=-1,
                        help='If it is non-negative, this script will draw detected and groundtruth'
                             'boxes')

    return parser


def main():
    """ Main function. """

    args = arg_parser().parse_args()
    config = load_config(args.config)

    if args.weights:
        print(args.weights, '(recall, precision, method_hmean)', test(args, config))
    elif args.weights_folder:
        args.weights_folder = os.path.abspath(args.weights_folder)

        try:
            with open(os.path.join(args.weights_folder, 'evaluations.txt')) as opened_file:
                already_tested_weights = [line.strip().split()[0] for line in opened_file.readlines()]
        except:
            already_tested_weights = []

        model = pixel_link_model(tf.keras.Input(shape=list(args.resolution)[::-1] + [3]), config)

        newly_tested = []
        with tf.summary.create_file_writer(
                os.path.join(args.weights_folder, '../logs')).as_default():

            weights_list = ['.'.join(x.split('.')[:-1])
                            for x in os.listdir(args.weights_folder) if x.startswith('model')]
            weights_list = list(set(weights_list))
            weights_list = [os.path.join(args.weights_folder, x) for x in weights_list]
            weights_list = [x for x in weights_list if x not in already_tested_weights]
            weights_list = sorted(weights_list, key=lambda x: parse_epoch(x))

            for weights in weights_list:
                args.weights = weights
                result = test(args, config, model=model)
                newly_tested.append(args.weights + str(result[-1]))
                already_tested_weights.append(args.weights)
                print(args.weights, '(recall, precision, method_hmean)', result)

                with open(os.path.join(args.weights_folder, 'evaluations.txt'), 'a+') as opened_file:
                    opened_file.write('{} {}\n'.format(args.weights, result[-1]))


if __name__ == '__main__':
    main()
