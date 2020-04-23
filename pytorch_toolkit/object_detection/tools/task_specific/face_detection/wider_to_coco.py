# Copyright (C) 2020 Intel Corporation
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

""" Converts WiderFace annotation to COCO format. """

# pylint: disable=R0914

import json
import os

import argparse
import imagesize
from tqdm import tqdm


def parse_wider_gt(ann_file):
    """ Parses wider annotation. """

    bboxes = dict()
    landmarks = dict()
    with open(ann_file) as read_file:
        content = [line.strip() for line in read_file.readlines()]
        new_file = True
        i = 0
        while True:
            if new_file:
                image_name = content[i]
                bboxes[image_name] = list()
                landmarks[image_name] = list()
                new_file = False
                i += 1
            else:
                bbox_num = int(content[i])
                if bbox_num == 0:
                    i += 1
                i += 1
                for _ in range(bbox_num):
                    xmin, ymin, width, height = [int(x) for x in content[i].split(' ')[:4]]
                    if width >= 0 and height >= 0:
                        bboxes[image_name].append([xmin, ymin, width, height])
                        landmarks[image_name].append([])
                    else:
                        print('Ignored because of invalid size: ', [xmin, ymin, width, height])
                    i += 1
                if i == len(content):
                    break
                new_file = True

    return bboxes, landmarks


def parse_wider_gt_with_landmarks(ann_file):
    """ Parses wider annotation with landmarks. """

    bboxes = dict()
    landmarks = dict()
    with open(ann_file) as read_file:
        content = [line.strip() for line in read_file.readlines()]
        new_file = True
        i = 0
        while True:
            if new_file:
                image_name = content[i][2:]
                bboxes[image_name] = list()
                landmarks[image_name] = list()
                new_file = False
                i += 1
            else:
                while True:
                    if i == len(content) or content[i].startswith('#'):
                        break
                    line_split = content[i].split(' ')
                    xmin, ymin, width, height = [int(x) for x in line_split[:4]]
                    if width >= 0 and height >= 0:
                        bboxes[image_name].append([xmin, ymin, width, height])
                        points = [float(x) if (i + 1) % 3 != 0 else float(x) + 1 for i, x in
                                  enumerate(line_split[4:-1])]
                        landmarks[image_name].append(points)
                    else:
                        print('Ignored because of invalid size: ', [xmin, ymin, width, height])
                    i += 1
                if i == len(content):
                    break
                new_file = True

    return bboxes, landmarks


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('input_annotation',
                        help="Path to annotation file like wider_face_train_bbx_gt.txt")
    parser.add_argument('images_dir',
                        help="Path to folder with images like WIDER_train/images")
    parser.add_argument('output_annotation', help="Path to output json file")
    parser.add_argument('--with_landmarks', action='store_true',
                        help="Whether to read landmarks")

    return parser.parse_args()


def convert_wider_annotation(ann_file, data_dir, out_file, with_landmarks):
    """ Converts wider annotation to COCO format. """

    img_id = 0
    ann_id = 0
    cat_id = 1

    ann_dict = {}
    categories = [{"id": 1, "name": 'face'}]
    images_info = []
    annotations = []

    if with_landmarks:
        boxes, landmarks = parse_wider_gt_with_landmarks(ann_file)
    else:
        boxes, landmarks = parse_wider_gt(ann_file)

    for filename in tqdm(boxes.keys()):
        image_info = {}
        image_info['id'] = img_id
        img_id += 1
        image_info['width'], image_info['height'] = imagesize.get(os.path.join(data_dir, filename))
        image_info['file_name'] = os.path.relpath(
            os.path.join(data_dir, filename), os.path.dirname(out_file))
        images_info.append(image_info)

        for gt_bbox, gt_landmarks in zip(boxes[filename], landmarks[filename]):
            ann = {
                'id': ann_id,
                'image_id': image_info['id'],
                'segmentation': [],
                'keypoints': gt_landmarks,
                'category_id': cat_id,
                'iscrowd': 0,
                'area': gt_bbox[2] * gt_bbox[3],
                'bbox': gt_bbox
            }
            ann_id += 1
            annotations.append(ann)

    ann_dict['images'] = images_info
    ann_dict['categories'] = categories
    ann_dict['annotations'] = annotations
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, 'w') as outfile:
        outfile.write(json.dumps(ann_dict))


def main():
    """ Main function. """

    args = parse_args()
    convert_wider_annotation(args.input_annotation, args.images_dir,
                             args.output_annotation, args.with_landmarks)


if __name__ == '__main__':
    main()
