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

from ote.tests.utils import run_tests_by_pattern


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='train_tests_*.py')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def main():
    pass


if __name__ == '__main__':
    main()
