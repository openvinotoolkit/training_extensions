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

""" This module contains unit tests. """

import argparse
import os
import sys

from common.utils import run_tests_by_pattern


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='export_tests_*.py')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def main():
    if os.path.abspath(os.getcwd()) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..')):
        return 0

    args = parse_args()

    was_successful = run_tests_by_pattern(folder=os.path.dirname(__file__),
                                          pattern=args.pattern,
                                          verbose=args.verbose)
    ret = not was_successful
    sys.exit(ret)


if __name__ == '__main__':
    main()
