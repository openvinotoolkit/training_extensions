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

import logging
import os
from subprocess import run
import unittest


class TestWorkaround(unittest.TestCase):

    def test_all(self):
        logging.warning('THIS IS WORKAROUND FOR TESTING.')
        commands = [
            f'cd {os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")}',
            'pip3 install -e ote',
            'python3 tests/run_model_templates_tests.py'
        ]
        returncode = run(';'.join(commands), shell=True, check=True).returncode
        self.assertEqual(returncode, 0)


if __name__ == '__main__':
    unittest.main()
