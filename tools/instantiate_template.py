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
import logging
import os

import yaml

from ote.utils.misc import download_snapshot_if_not_yet, run_through_shell


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('template', help='Location of model template file (template.yaml).')
    parser.add_argument('output', help='Location of output directory where template will be instantiated.')
    parser.add_argument('--do-not-load-snapshot', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='If the instantiation should be run in verbose mode')

    return parser.parse_args()


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    with open(args.template) as read_file:
        content = yaml.load(read_file, yaml.SafeLoader)

    os.makedirs(args.output, exist_ok=True)
    run_through_shell(f'cp -r {os.path.dirname(args.template)}/* --target-directory={args.output}',
                      verbose=args.verbose)

    for dependency in content['dependencies']:
        source = dependency['source']
        destination = dependency['destination']
        if destination != 'snapshot.pth':
            rel_source = os.path.join(os.path.dirname(args.template), source)
            cur_dst = os.path.join(args.output, destination)
            os.makedirs(os.path.dirname(cur_dst), exist_ok=True)
            run_through_shell(f'cp -r --no-target-directory {rel_source} {cur_dst}', check=True,
                              verbose=args.verbose)

    if not args.do_not_load_snapshot:
        download_snapshot_if_not_yet(args.template, args.output)


if __name__ == '__main__':
    main()
