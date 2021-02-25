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

""" This module allows you to create and write tf record dataset. """

import argparse
import os

from text_detection.annotation import write_to_tfrecords


def parse_args():
    """ Parses arguments. """

    args = argparse.ArgumentParser()
    args.add_argument('--input_datasets', required=True, help='Comma-separated datasets paths.')
    args.add_argument('--output', required=True, help='Path where output tf record will be written to.')
    args.add_argument('--imshow_delay', type=int, default=-1,
                      help='If it is non-negative, this script will draw detected and groundtruth boxes')

    return args.parse_args()


def main():
    """ Main function. """

    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_to_tfrecords(output_path=args.output, datasets=args.input_datasets.split(','),
                       imshow_delay=args.imshow_delay)


if __name__ == '__main__':
    main()
