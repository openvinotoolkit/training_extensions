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
import unittest
from subprocess import run

def run_tests_by_pattern(folder, pattern, verbose):
    logging.basicConfig(level=logging.INFO)
    if verbose:
        verbosity=2
    else:
        verbosity=1
    testsuite = unittest.TestLoader().discover(folder, pattern=pattern)
    was_successful = unittest.TextTestRunner(verbosity=verbosity).run(testsuite).wasSuccessful()
    return was_successful
