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

from os import makedirs
from os.path import exists, join, basename
from shutil import rmtree
from argparse import ArgumentParser

import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from json import dump as json_dump


def parse_annotation(annot_path, input_height, input_width):
    """Loads annotation from the .mat file.

    :param annot_path: Path to annotation file
    :param input_height: Input image height
    :param input_width: Input image width
    :return: Dict with annotation
    """

    mat = loadmat(annot_path)

    annot_header_name = [header for header in list(mat) if header.startswith('anno')][0]
    in_data = mat[annot_header_name][0]

    num_images = len(in_data)

    out_data = {}
    for image_id in xrange(num_images):
        image_data = in_data[image_id][0, 0]

        image_name = image_data[1][0]
        in_detections = image_data[2]

        out_detections = [{'label': int(det[0]),
                           'xmin': float(det[1]) / float(input_width),
                           'ymin': float(det[2]) / float(input_height),
                           'xmax': float(det[1] + det[3]) / float(input_width),
                           'ymax': float(det[2] + det[4]) / float(input_height),
                           'track_id': int(det[5]),
                           'occluded': False}
                                for det in in_detections]

        filtered_detections = [det for det in out_detections if det['label'] not in [0, 5]]
        if len(filtered_detections) == 0:
            continue

        out_data[image_name] = filtered_detections

    return out_data


def create_dir(dir_path):
    """Creates directory if needed.

    :param dir_path: Path to new directory
    """

    if exists(dir_path):
        rmtree(dir_path)

    makedirs(dir_path)


def save_data_paths(data, out_path):
    """Saves paths to data into specified file.

    :param data: data to save
    :param out_path: Path to save
    """

    with open(out_path, 'w') as input_stream:
        for image_path, annot_path in tqdm(data, desc='Dumping image paths'):
            input_stream.write('{} {}\n'.format(image_path, annot_path))


def main():
    """Main function.
    """

    parser = ArgumentParser()
    parser.add_argument('--images_dir', '-i', type=str, required=True, help='Path to directory with images')
    parser.add_argument('--annot', '-a', type=str, required=True, help='Path to .mat file with annotation')
    parser.add_argument('--out_dir', '-o', type=str, required=True, help='Path to save the converted annotation')
    parser.add_argument('--input_size', type=str, required=False, default='1024,2048', help='Input image size: HxW')
    args = parser.parse_args()

    assert exists(args.images_dir)
    assert exists(args.annot)

    input_height, input_width = map(int, args.input_size.split(','))

    annotation_dir = join(args.out_dir, 'annotation')
    images_dir = join(args.out_dir, 'images')

    create_dir(annotation_dir)
    create_dir(images_dir)

    annotation = parse_annotation(args.annot, input_height, input_width)

    dumped_paths = []
    total_num_bboxes = 0
    max_num_bboxes = 0
    for image_name in tqdm(annotation, desc='Converting'):
        detections = annotation[image_name]

        file_name, extension = basename(image_name).split('.')
        city_name = file_name.split('_')[0]

        annot_path = join(annotation_dir, '{}.json'.format(file_name))
        with open(annot_path, 'w') as out_stream:
            json_dump(detections, out_stream)

        in_image_path = join(args.images_dir, city_name, image_name)
        src_bgr_image = cv2.imread(in_image_path, cv2.IMREAD_COLOR)
        grayscale_image = cv2.cvtColor(src_bgr_image, cv2.COLOR_BGR2GRAY)

        rgb_image = np.tile(grayscale_image.reshape(input_height, input_width, 1), (1, 1, 3))

        out_image_path = join(images_dir, image_name)
        cv2.imwrite(out_image_path, rgb_image)

        dumped_paths.append((out_image_path, annot_path))
        total_num_bboxes += len(detections)
        max_num_bboxes = max(max_num_bboxes, len(detections))

    out_data_path = join(args.out_dir, 'data.txt')
    save_data_paths(dumped_paths, out_data_path)
    print('\nLoaded frames: {} with {} boxes. Max number bboxes on image: {}'
          .format(len(annotation), total_num_bboxes, max_num_bboxes))


if __name__ == '__main__':
    main()
