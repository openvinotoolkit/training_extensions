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

# pylint: disable=W1203,C0411,C0413,no-value-for-parameter

import argparse
import logging
import os
import sys

from mmcv.utils import Config
import yaml

from eval import main as evaluate

sys.path.append(f'{os.path.abspath(os.path.dirname(__file__))}/../../')
from tools.misc import train, get_work_dir


def parse_args():
    """ Parses input args. """

    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='A path to model training configuration file (.py).')
    parser.add_argument('gpu_num', type=int,
                        help='A number of GPUs to use in training.')
    parser.add_argument('out',
                        help='A path to output file where models metrics will be saved (.yml).')
    parser.add_argument('--update_config',
                        help='Update configuration file by parameters specified here.'
                             'Use quotes if you are going to change several params.',
                        default='')
    parser.add_argument('--show-dir', '--show_dir', dest='show_dir',
                        help='A directory where images with drawn detected objects will be saved.')

    return parser.parse_args()


def main():
    """ Main function. """
    logging.basicConfig(level=logging.DEBUG)

    args = parse_args()
    logging.info(f'Commandline:\n{" ".join(sys.argv)}')

    cfg = Config.fromfile(args.config)

    update_config = f' --update_config {args.update_config}' if args.update_config else ''

    logging.info('Training started ...')
    training_info = train(args.config, args.gpu_num, update_config)
    logging.info('... training completed.')
    work_dir = get_work_dir(cfg, args.update_config)

    logging.info('Evaluation started ...')
    evaluate(os.path.join(work_dir, "config.py"), os.path.join(work_dir, "latest.pth"), args.out, '', args.show_dir)
    logging.info('... evaluation completed.')

    with open(args.out, 'a+') as dst_file:
        yaml.dump(training_info, dst_file)


if __name__ == '__main__':
    main()
