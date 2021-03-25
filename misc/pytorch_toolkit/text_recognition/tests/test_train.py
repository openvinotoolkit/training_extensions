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
from tempfile import mkdtemp

from text_recognition.utils.trainer import Trainer
from text_recognition.utils.get_config import get_config


def create_train_test(config_file):

    class TestTrain(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            train_config = get_config(config_file, section='train')
            cls.config = train_config
            cls.config['epochs'] = 1
            cls.config['_test_steps'] = 40
            cls.work_dir = mkdtemp()
            cls.trainer = Trainer(work_dir=cls.work_dir, config=cls.config)

        def test_train(self):
            self.trainer.train()
            cur_loss = self.trainer.current_loss
            self.trainer.train()
            self.assertLessEqual(self.trainer.current_loss, cur_loss)
    return TestTrain


class TestMediumRenderedTrain(create_train_test('configs/medium_config.yml')):
    'Test case for medium config'


class TestHandwrittenPolynomialsTrain(create_train_test('configs/polynomials_handwritten_config.yml')):
    'Test case for handwritten polynomials config'


class TestAlphanumericTrain(create_train_test('configs/config_0013.yml')):
    'Test case for alphanumeric config'


if __name__ == '__main__':
    unittest.main()
