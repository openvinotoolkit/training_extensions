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

import os
from random import shuffle, seed
import argparse

SEED = 33

def parse_args():
  parser = argparse.ArgumentParser(description='Make train/val split of original annotation.')
  parser.add_argument('path_to_annotation', help='Path to Synthetic Chinese License Plates annotation file.')
  return parser.parse_args()

def main():
  seed(SEED)
  args = parse_args()

  annotation_dir = os.path.dirname(os.path.realpath(args.path_to_annotation))
  with open(args.path_to_annotation) as f:
    annotations = [line.split() for line in f]

  shuffle(annotations)
  train_len = int(len(annotations) * 0.99)
  val_len = len(annotations) - train_len

  with open(os.path.join(annotation_dir, 'train'), 'w') as f:
    for line in annotations[:train_len]:
      f.write(os.path.join(annotation_dir, line[0]) + ' ' + line[1] + '\n')

  with open(os.path.join(annotation_dir, 'val'), 'w') as f:
    for line in annotations[train_len:]:
      f.write(os.path.join(annotation_dir, line[0]) + ' ' + line[1] + '\n')

  with open(os.path.join(annotation_dir, 'test_infer'), 'w') as f:
    for line in annotations[train_len:]:
      f.write(os.path.join(annotation_dir, line[0]) + '\n')

if __name__ == '__main__':
  main()
