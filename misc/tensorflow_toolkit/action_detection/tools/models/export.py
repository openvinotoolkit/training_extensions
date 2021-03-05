#!/usr/bin/env python2
#
# Copyright (C) 2019 Intel Corporation
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

from os import makedirs
from os.path import exists, basename, join
from argparse import ArgumentParser

from action_detection.nn.monitors.factory import get_monitor

BASE_FILE_NAME = 'converted_model'
CKPT_FILE_NAME = '{}.ckpt'.format(BASE_FILE_NAME)
PB_FILE_NAME = '{}.pbtxt'.format(BASE_FILE_NAME)
FROZEN_FILE_NAME = 'frozen.pb'


def main():
    """Carry out model preparation for the export.
    """

    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    parser.add_argument('--snapshot_path', '-s', type=str, required=True, default='', help='Path to model snapshot')
    parser.add_argument('--output_dir', '-o', type=str, required=True, default='', help='Path to output directory')
    args = parser.parse_args()

    assert exists(args.config)
    assert exists(args.snapshot_path + '.index')

    if not exists(args.output_dir):
        makedirs(args.output_dir)

    task_monitor = get_monitor(args.config, snapshot_path=args.snapshot_path)

    converted_snapshot_path = join(args.output_dir, CKPT_FILE_NAME)
    task_monitor.eliminate_train_ops(converted_snapshot_path)

    converted_model_path = '{}-{}'.format(converted_snapshot_path,
                                          int(basename(args.snapshot_path).split('-')[-1]))
    task_monitor.save_model_graph(converted_model_path, args.output_dir)

    task_monitor.freeze_model_graph(converted_model_path,
                                    join(args.output_dir, PB_FILE_NAME),
                                    join(args.output_dir, FROZEN_FILE_NAME))

if __name__ == '__main__':
    main()
