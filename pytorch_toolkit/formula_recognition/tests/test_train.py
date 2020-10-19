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

import os
import shutil
import unittest

import yaml
from tools.train import Trainer

CONFIGS = [
    'configs/medium_config.yml',
    'configs/polynomials_handwritten_config.yml'
]


def create_train_test(config_file):

    class TestTrain(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            with open(config_file, 'r') as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
                train_config = config.get("train")
                common_config = config.get("common")
                train_config.update(common_config)
            cls.config = train_config
            cls.config['epochs'] = 1
            cls.config['_test_steps'] = 40
            cls.trainer = Trainer(work_dir='./..', config=cls.config)

        def test_train(self):
            self.trainer.train()
            cur_loss = self.trainer._current_loss
            self.trainer.train()
            self.assertLessEqual(self.trainer._current_loss, cur_loss)
            if os.path.exists(self.trainer.logs_path):
                shutil.rmtree(self.trainer.logs_path)
    return TestTrain


class TestMediumRenderedTrain(create_train_test(CONFIGS[0])):
    "Test case for medium config"


class TestHandwrittenPolynomialsTrain(create_train_test(CONFIGS[0])):
    "Test case for handwritten polynomials config"

if __name__ == "__main__":
    unittest.main()
