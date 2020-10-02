"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest

import numpy as np
import yaml

from tools.test import Evaluator


class TestEval(unittest.TestCase):
    def setUp(self):
        with open('configs/config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader).get("eval")
        self.config = config
        self.validator = Evaluator(config=self.config)

    def test_validate(self):
        metric = self.validator.validate()
        self.assertGreaterEqual(metric, self.config.get("target_metric"))


if __name__ == "__main__":
    unittest.main()
