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
import os

from text_spotting.datasets.datasets import TextOnlyCocoAnnotation
from text_spotting.datasets.datasets import str_to_class
from text_spotting.datasets.factory import root_data_dir


def parse_args():
    """ Parses input arguments. """

    args = argparse.ArgumentParser()
    args.add_argument('--config', help='Path to dataset configuration file (json).',
                      required=True)
    args.add_argument('--output', help='Path where to save annotation (json).',
                      required=True)
    args.add_argument('--visualize', action='store_true', help='Visualize annotation.')
    return args.parse_args()


def main():
    """ Loads configuration file and creates dataset. """

    args = parse_args()
    with open(args.config) as file:
        config = json.load(file)

    assert isinstance(config, list)
    ann = TextOnlyCocoAnnotation()
    for dataset in config:
        assert isinstance(dataset, dict)
        if os.path.islink(root_data_dir()):
            dataset['kwargs']['root'] = os.readlink(root_data_dir())
        else:
            dataset['kwargs']['root'] = os.path.abspath(root_data_dir())
        ann += str_to_class[dataset['name']](**dataset['kwargs'])()

    ann.write(args.output)

    ann = TextOnlyCocoAnnotation(args.output, os.path.dirname(args.output))
    if args.visualize:
        ann.visualize(put_text=True, imshow_delay=1)


if __name__ == '__main__':
    main()
