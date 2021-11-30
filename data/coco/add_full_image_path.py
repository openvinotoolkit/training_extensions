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

import argparse
import json
import os

from tqdm import tqdm


def parse_args():
  example_text = '''Usage example:

   python add_full_image_path.py instances_train2017.json ~/coco/train2017 instances_train2017_full_paths.json
   '''

  parser = argparse.ArgumentParser(description='Add full image paths to an annotation in COCO format',
                                   epilog=example_text,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('annotation_path', help='Path to an annotation in COCO format')
  parser.add_argument('images_dir', help='Path to a directory with images')
  parser.add_argument('output_path', help='Path to an output file')
  return parser.parse_args()


def add_full_path(annotation_path, images_dir, output_path):
  with open(annotation_path) as f:
    data = json.load(f)

  for im in tqdm(data['images'], total=len(data['images'])):
    im['image'] = os.path.join(images_dir, im['file_name'])

  with open(output_path, 'w') as f:
    json.dump(data, f)


def main():
  args = parse_args()
  add_full_path(args.annotation_path, args.images_dir, args.output_path)


if __name__ == '__main__':
  main()
