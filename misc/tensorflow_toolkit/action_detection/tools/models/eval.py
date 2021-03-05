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

from os.path import exists
from argparse import ArgumentParser

from action_detection.nn.monitors.factory import get_monitor


def main():
    """Carry out model evaluation.
    """

    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    parser.add_argument('--val_data', '-v', type=str, required=True, help='Path to file with annotated train images')
    parser.add_argument('--snapshot_path', '-s', type=str, required=False, default='', help='Path to snapshot')
    parser.add_argument('--batch_size', '-b', type=int, required=False, default=8, help='Batch size')
    args = parser.parse_args()

    assert exists(args.config)
    assert exists(args.val_data)
    assert exists(args.snapshot_path + '.index')
    assert args.batch_size > 0

    task_monitor = get_monitor(args.config, args.batch_size, snapshot_path=args.snapshot_path)
    task_monitor.test(args.val_data)

if __name__ == '__main__':
    main()
