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
from os.path import exists
from argparse import ArgumentParser

from action_detection.nn.monitors.factory import get_monitor


def main():
    """Carry out model training.
    """

    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    parser.add_argument('--train_data', '-t', type=str, required=True, help='Path to file with annotated train images')
    parser.add_argument('--log_dir', '-l', type=str, required=True, help='Path to save logs')
    parser.add_argument('--snapshot_path', '-s', type=str, required=False, default='', help='Path to snapshot')
    parser.add_argument('--init_model_path', '-i', type=str, required=False, default='', help='Path to init weights')
    parser.add_argument('--src_scope', type=str, required=False, default='', help='Checkpoint scope name')
    parser.add_argument('--batch_size', '-b', type=int, required=False, default=8, help='Batch size')
    parser.add_argument('--num_gpu', '-n', type=int, required=False, default=1, help='Number of GPUs')
    args = parser.parse_args()

    assert exists(args.config)
    assert exists(args.train_data)
    assert args.batch_size > 0
    assert args.num_gpu > 0

    if not exists(args.log_dir):
        makedirs(args.log_dir)

    task_monitor = get_monitor(args.config, args.batch_size, args.num_gpu, args.log_dir,
                               args.src_scope, args.snapshot_path, args.init_model_path)
    task_monitor.train(args.train_data)

if __name__ == '__main__':
    main()
