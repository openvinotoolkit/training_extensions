#!/usr/bin/env python2
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

from __future__ import print_function

from argparse import ArgumentParser
from os.path import exists

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def parse_data(data_path):
    with open(data_path, 'r') as input_stream:
        out_data = []
        for line in input_stream:
            if line.endswith('\n'):
                line = line[:-len('\n')]

            if len(line) == 0:
                continue

            image_path, annotation_path = line.split(' ')

            if not exists(image_path) or not exists(annotation_path):
                continue

            out_data.append((image_path, annotation_path))

    return out_data


def draw_actions(image, bboxes):
    image = np.copy(image)
    image_height, image_width = image.shape[:2]

    if len(bboxes) == 0:
        return image

    for bbox in bboxes:
        ymin = int(bbox['ymin'] * image_height)
        xmin = int(bbox['xmin'] * image_width)
        ymax = int(bbox['ymax'] * image_height)
        xmax = int(bbox['xmax'] * image_width)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return image


def main():
    """Main function to dump frames for the specified tasks.
    """

    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=str, required=True, help='Path to the file with tasks')
    args = parser.parse_args()

    assert exists(args.data)

    data_paths = parse_data(args.data)

    for pair_id in trange(len(data_paths)):
        image_path, annotation_path = data_paths[pair_id]

        image = cv2.imread(image_path)

        with open(annotation_path, 'r') as read_file:
            annotation = json.load(read_file)

        image_with_annotation = draw_actions(image, annotation)

        plt.imshow(image_with_annotation)
        plt.show()


if __name__ == '__main__':
    main()
