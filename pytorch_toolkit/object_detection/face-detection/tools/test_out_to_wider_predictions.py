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

""" This script converts output of test.py (mmdetection) to a set of files
that can be passed to official WiderFace evaluation procedure."""

import argparse
import os
from tqdm import tqdm

import mmcv

from mmdet.datasets import build_dataset


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser(
        description='This script converts output of test.py (mmdetection) to '
                    'a set of files that can be passed to official WiderFace '
                    'evaluation procedure.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('input', help='output result file from test.py')
    parser.add_argument('out_folder', help='folder where to store WiderFace '
                                           'evaluation-friendly output')
    args = parser.parse_args()
    return args


def main():
    """ Main function. """

    args = parse_args()

    if args.input is not None and not args.input.endswith(('.pkl', '.pickle')):
        raise ValueError('The input file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)

    results = mmcv.load(args.input)

    wider_friendly_results = []
    for i, sample in enumerate(tqdm(dataset)):
        filename = sample['img_meta'][0].data['filename']
        folder, image_name = filename.split('/')[-2:]
        wider_friendly_results.append({'folder': folder, 'name': image_name[:-4],
                                       'boxes': results[i][0]})

    for result in wider_friendly_results:
        folder = os.path.join(args.out_folder, result['folder'])
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, result['name'] + '.txt'), 'w') as write_file:
            write_file.write(result['name'] + '\n')
            write_file.write(str(len(result['boxes'])) + '\n')
            for box in result['boxes']:
                box = box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]
                write_file.write(' '.join([str(x) for x in box]) + '\n')


if __name__ == '__main__':
    main()
