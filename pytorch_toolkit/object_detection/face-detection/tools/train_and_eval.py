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

# pylint: disable=C0301,W0622,R0914

import argparse
import subprocess
import os

from mmcv.utils import Config

from eval import eval


def parse_args():
    """ Parses input args. """

    args = argparse.ArgumentParser()
    args.add_argument('config',
                      help='A path to model training configuration file (.py).')
    args.add_argument('gpu_num',
                      help='A number of GPU to use in training.')
    args.add_argument('out',
                      help='A path to output file where models metrics will be saved (.yml).')
    args.add_argument('--wider_dir',
                      help='Specify this  path if you would like to test your model on WiderFace dataset.')

    return args.parse_args()


def main():
    """ Main function. """

    args = parse_args()

    if args.wider_dir:
        wider_val_zip = os.path.join(args.wider_dir, 'WIDER_val.zip')
        assert os.path.exists(wider_val_zip), f'failed to find WIDER_val.zip here: {wider_val_zip}'

    mmdetection_tools = f'{os.path.dirname(__file__)}/../../../../external/mmdetection/tools'

    subprocess.run(f'{mmdetection_tools}/dist_train.sh'
                   f' {args.config}'
                   f' {args.gpu_num}'.split(' '), check=True)

    cfg = Config.fromfile(args.config)

    eval(args.config, os.path.join(cfg.work_dir, "latest.pth"), args.wider_dir, args.out)


if __name__ == '__main__':
    main()
