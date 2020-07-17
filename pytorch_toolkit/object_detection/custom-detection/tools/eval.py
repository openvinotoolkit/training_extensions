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

import argparse
import sys
sys.path.append(f'{os.path.abspath(os.path.dirname(__file__))}/../../')

from tools.misc import evaluate, coco_ap_eval


def parse_args():
    """ Parses input args. """

    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='A path to model training configuration file (.py).')
    parser.add_argument('snapshot',
                        help='A path to pre-trained snapshot (.pth).')
    parser.add_argument('out',
                        help='A path to output file where models metrics will be saved (.yml).')
    parser.add_argument('--update_config',
                        help='Update configuration file by parameters specified here.'
                             'Use quotes if you are going to change several params.',
                        default='')

    return parser.parse_args()


def main(config, snapshot, out, update_config):
    """ Main function. """

    metrics_functions = [coco_ap_eval]
    evaluate(config, snapshot, out, update_config, metrics_functions)


if __name__ == '__main__':
    args = parse_args()
    main(args.config, args.snapshot, args.out, args.update_config)
