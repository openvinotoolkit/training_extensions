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

import logging
import os
import unittest
from subprocess import run


def collect_ap(path):
    ap = []
    beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file.readlines()]
        for line in content:
            if line.startswith(beginning):
                ap.append(float(line.replace(beginning, '')))
    return ap


def download_if_not_yet(output_folder, url):
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, os.path.basename(url))
    if not os.path.exists(path):
        os.system(f'wget --no-verbose {url} -P {output_folder}')
    return path


def relative_abs_error(expected, actual):
    return abs(expected - actual) / expected


def run_through_shell(cmd):
    cmdstr = ' '.join(cmd) if isinstance(cmd, list) else cmd
    cmdstr = cmdstr.replace(';', ';\n').replace(' --', ' \\\n    --')
    logging.info(f'Running through shell cmd\n`{cmdstr}\n`')
    run(cmd, shell=True, check=True, executable="/bin/bash")

def run_tests_by_pattern(folder, pattern, verbose):
    logging.basicConfig(level=logging.INFO)
    if verbose:
        verbosity = 2
    else:
        verbosity = 1
    testsuite = unittest.TestLoader().discover(folder, pattern=pattern)
    was_successful = unittest.TextTestRunner(verbosity=verbosity).run(testsuite).wasSuccessful()
    return was_successful
