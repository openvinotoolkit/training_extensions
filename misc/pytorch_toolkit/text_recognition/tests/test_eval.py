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
import os.path

from text_recognition.utils.evaluator import Evaluator
from text_recognition.utils.get_config import get_config
from text_recognition.utils.common import download_checkpoint


def create_evaluation_test_case(config_file, expected_outputs):

    class TestEval(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            test_config = get_config(config_file, section='eval')
            cls.config = test_config
            cls.config.update({'expected_outputs': expected_outputs})
            if not os.path.exists(cls.config.get("model_path")):
                download_checkpoint(cls.config.get("model_path"), cls.config.get("model_url"))
            cls.validator = Evaluator(config=cls.config)

        def test_validate(self):
            metric = self.validator.validate()
            target_metric = self.validator.expected_outputs.get('target_metric')
            self.assertGreaterEqual(metric, target_metric)
    return TestEval


class TestMediumRenderedEvaluation(
        create_evaluation_test_case(
            'configs/medium_config.yml',
            expected_outputs='tests/expected_outputs/formula_recognition/medium_photographed_0185.json')):
    'Test case for medium config'


class TestHandwrittenPolynomialsEvaluation(
        create_evaluation_test_case(
            'configs/polynomials_handwritten_config.yml',
            expected_outputs='tests/expected_outputs/formula_recognition/polynomials_handwritten_0166.json')):
    'Test case for handwritten polynomials config'


class TestAlphanumeric0014Evaluation(
        create_evaluation_test_case(
            'configs/config_0014.yml',
            expected_outputs='tests/expected_outputs/alphanumeric/icdar13_greater3_0014.json')):
    'Test case for alphanumeric config'


class TestAlphanumeric0015Evaluation(
        create_evaluation_test_case(
            'configs/config_0015.yml',
            expected_outputs='tests/expected_outputs/alphanumeric/icdar13_greater3_0015.json')):
    'Test case for alphanumeric text recognition config'


class TestAlphanumeric0016Evaluation(
        create_evaluation_test_case(
            'configs/config_0016.yml',
            expected_outputs='tests/expected_outputs/alphanumeric/icdar13_greater3_0016.json')):
    'Test case for alphanumeric text recognition config'


if __name__ == '__main__':
    unittest.main()
