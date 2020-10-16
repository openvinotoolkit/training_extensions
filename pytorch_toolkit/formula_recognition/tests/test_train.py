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
import shutil
import os

import yaml

from tools.train import Trainer


class TestTrain(unittest.TestCase):
    def setUp(self):
        with open('configs/config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            train_config = config.get("train")
            common_config = config.get("common")
            train_config.update(common_config)
        self.config = train_config
        self.config['epochs'] = 1
        self.config['_test_steps'] = 20
        self.trainer = Trainer(work_dir='./..', config=self.config)

    def test_train(self):
        self.trainer.train()
        cur_loss = self.trainer._current_loss
        self.trainer.train()
        self.assertLess(self.trainer._current_loss, cur_loss)
        if os.path.exists(self.trainer.logs_path):
            shutil.rmtree(self.trainer.logs_path)


if __name__ == "__main__":
    unittest.main()
