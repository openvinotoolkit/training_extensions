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

from __future__ import print_function

from os.path import exists
from argparse import ArgumentParser

from action_detection.nn.monitors.factory import get_monitor


def main():
    """Carry out model demonstration.
    """

    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to video')
    parser.add_argument('--snapshot_path', '-s', type=str, required=False, default='', help='Path to snapshot')
    parser.add_argument('--out_scale', type=float, default=1.0, help='Output frame scale')
    parser.add_argument('--deploy', '-d', action='store_true', help='Execute in deploy mode')
    args = parser.parse_args()

    assert exists(args.config)
    assert exists(args.input)
    assert exists(args.snapshot_path + '.index')
    assert args.out_scale > 0.0

    task_monitor = get_monitor(args.config, snapshot_path=args.snapshot_path)
    task_monitor.demo(args.input, args.out_scale, args.deploy)

if __name__ == '__main__':
    main()
