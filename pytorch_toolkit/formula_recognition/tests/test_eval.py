import unittest

import numpy as np
import yaml

from tools.test import Evaluator


class TestEval(unittest.TestCase):
    def setUp(self):
        with open('configs/eval_config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = config
        self.validator = Evaluator(config=self.config)

    def test_validate(self):
        metric = self.validator.validate()
        self.assertEqual(metric, 1.0)


if __name__ == "__main__":
    unittest.main()
