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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def parse_args():
    """ Parses input arguments. """

    args = argparse.ArgumentParser()
    args.add_argument('--coco_annotation_json', required=True)
    args.add_argument('--num_clusters', type=int, required=True)
    args.add_argument('--image_size', type=int, nargs=2, required=True)

    return args.parse_args()


def main():
    """ Clusters bboxes. """

    args = parse_args()

    with open(args.coco_annotation_json) as file:
        annotation = json.load(file)

    images_id_to_size = dict()
    for image_info in annotation['images']:
        images_id_to_size[image_info['id']] = {
            'width': image_info['width'],
            'height': image_info['height']
        }

    widths = []
    heights = []

    for bbox_info in annotation['annotations']:
        bbox_width = bbox_info['bbox'][2] / images_id_to_size[bbox_info['image_id']]['width']
        bbox_height = bbox_info['bbox'][3] / images_id_to_size[bbox_info['image_id']]['height']

        widths.append(bbox_width)
        heights.append(bbox_height)

    sizes = np.array([widths, heights]).transpose()

    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(sizes)
    centers = kmeans.cluster_centers_.copy()

    plt.scatter(widths, heights)
    plt.scatter(centers[:, 0], centers[:, 1])

    centers *= args.image_size
    centers = sorted(centers, key=lambda x: x[0] * x[1])

    print('widths', ', '.join([str(c[0]) for c in centers]))
    print('heights', ', '.join([str(c[1]) for c in centers]))

    plt.show()


if __name__ == '__main__':
    main()
