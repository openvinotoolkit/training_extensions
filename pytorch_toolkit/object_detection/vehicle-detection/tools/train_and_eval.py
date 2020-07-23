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

# pylint: disable=C0411,C0413,no-value-for-parameter

import argparse
import os
import sys

from mmcv.utils import Config
import yaml
import torch

from eval import main as evaluate
sys.path.append(f'{os.path.abspath(os.path.dirname(__file__))}/../../')
from tools.misc import run_with_termination

def parse_args():
    """ Parses input args. """

    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='A path to model training configuration file (.py).')
    parser.add_argument('gpu_num', type=int,
                        help='A number of GPU to use in training.')
    parser.add_argument('out',
                        help='A path to output file where models metrics will be saved (.yml).')
    parser.add_argument('--update_config',
                        help='Update configuration file by parameters specified here.'
                             'Use quotes if you are going to change several params.',
                        default='')

    return parser.parse_args()


def main():
    """ Main function. """

    args = parse_args()
    print(sys.argv)
    sys.stdout.flush()

    mmdetection_tools = f'{os.path.dirname(__file__)}/../../../../external/mmdetection/tools'

    cfg = Config.fromfile(args.config)

    update_config = f' --update_config {args.update_config}' if args.update_config else ''

    if torch.cuda.is_available():
        available_gpu_num = torch.cuda.device_count()
        gpu_num = args.gpu_num
        if available_gpu_num < args.gpu_num:
            print(f'available_gpu_num < args.gpu_num: {available_gpu_num} < {args.gpu_num}')
            print(f'decreased number of gpu to: {available_gpu_num}')
            gpu_num = available_gpu_num
            sys.stdout.flush()
        run_with_termination(f'{mmdetection_tools}/dist_train.sh'
                             f' {args.config}'
                             f' {gpu_num}'
                             f'{update_config}'.split(' '))
    else:
        run_with_termination(f'python {mmdetection_tools}/train.py'
                             f' {args.config}'
                             f'{update_config}'.split(' '))

    overridden_work_dir = [p.split('=') for p in args.update_config.strip().split(' ') if
                           p.startswith('work_dir=')]
    if overridden_work_dir:
        cfg.work_dir = overridden_work_dir[0][1]

    evaluate(os.path.join(cfg.work_dir, "config.py"), os.path.join(cfg.work_dir, "latest.pth"), args.out, '')

    with open(args.out, 'a+') as dst_file:
        yaml.dump({'training_gpu_num': args.gpu_num}, dst_file)


if __name__ == '__main__':
    main()
