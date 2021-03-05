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

""" This module allows you to convert source dataset to internal format. """

import argparse
from text_detection.annotation import TextDetectionDataset


def parse_args():
    """ Parses input arguments. """

    args = argparse.ArgumentParser()
    args.add_argument('--out_annotation', help='Path where annotaion will be saved to.',
                      required=True)
    args.add_argument('--in_annotation', help='Path to annotation in source format.')
    args.add_argument('--images', help='Path to dataset images.', required=True)
    args.add_argument('--type', choices=['icdar15', 'toy', 'icdar17_mlt', 'icdar19_mlt', 'cocotext_v2'],
                      help='Source dataset type/name.', required=True)
    args.add_argument('--train', action='store_true')
    args.add_argument('--imshow_delay', type=int, default=-1,
                      help='If it is non-negative, this script will draw detected and groundtruth'
                           'boxes')

    return args.parse_args()


def main():
    """ Main function. """
    args = parse_args()

    if args.type == 'icdar15':
        text_detection_dataset = TextDetectionDataset.read_from_icdar2015(
            args.images, args.annotation, is_training=args.train)
    elif args.type == 'icdar19_mlt':
        text_detection_dataset = TextDetectionDataset.read_from_icdar2019_mlt(args.images)
    elif args.type == 'icdar17_mlt':
        text_detection_dataset = TextDetectionDataset.read_from_icdar2017_mlt(args.images, is_training=False)
    elif args.type == 'cocotext_v2':
        text_detection_dataset = TextDetectionDataset.read_from_coco_text(args.images)
    elif args.type == 'toy':
        text_detection_dataset = TextDetectionDataset.read_from_toy_dataset(args.images)

    text_detection_dataset.write(args.out_annotation)
    if args.imshow_delay >= 0:
        text_detection_dataset.visualize(put_text=True, imshow_delay=args.imshow_delay)


if __name__ == '__main__':
    main()
