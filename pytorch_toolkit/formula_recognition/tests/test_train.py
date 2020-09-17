import unittest

import numpy as np
import yaml

from tools.train import Trainer


class TestTrain(unittest.TestCase):
    def setUp(self):
        with open('configs/train_config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = config
        self.config['epochs'] = 1
        self.config['_test_steps'] = 20
        self.trainer = Trainer(work_dir='./', config=self.config)

    def test_train(self):
        self.trainer.train()
        cur_loss = self.trainer._current_loss
        self.trainer.train()
        self.assertLess(self.trainer._current_loss, cur_loss)


if __name__ == "__main__":
    unittest.main()
