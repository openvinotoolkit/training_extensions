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

import yaml
from tools.test import Evaluator

CONFIGS = [
    'configs/medium_config.yml',
    'configs/polynomials_handwritten_config.yml'
]


def create_evaluation_test_case(config_file):

    class TestEval(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            with open(config_file, 'r') as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
                test_config = config.get("eval")
                common_config = config.get("common")
                test_config.update(common_config)
            cls.config = test_config
            cls.validator = Evaluator(config=cls.config)

        def test_validate(self):
            metric = self.validator.validate()
            self.assertGreaterEqual(metric, self.config.get("target_metric"))
    return TestEval


class TestMediumRenderedEvaluation(create_evaluation_test_case(CONFIGS[0])):
    "Test case for medium config"


class TestHandwrittenPolynomialsEvaluation(create_evaluation_test_case(CONFIGS[0])):
    "Test case for handwritten polynomials config"


if __name__ == "__main__":
    unittest.main()
