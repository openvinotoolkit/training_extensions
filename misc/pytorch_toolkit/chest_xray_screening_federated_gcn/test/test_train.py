import unittest
import os
import sys
sys.path.append(r'/storage/adityak/federated/')
from src.utils.train_utils import train_model
from src.utils.downloader import download_checkpoint, download_data
from src.utils.get_config import get_config
from src.utils.train_utils import train_model

def create_train_test_for_without_gnn():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train', gnn=False)
            cls.config = config
            if not os.path.exists(config["data"]):
                download_data(gnn=True)

        def test_trainer(self):
            if not os.path.exists(self.config["checkpoint"]):
                download_checkpoint(gnn=False)
            train_model(self.config)

        def test_config(self):
            self.config = get_config(action='train', gnn=False)

    return TrainerTest

def create_train_test_for_with_gnn():
    class TrainerTestEff(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train', gnn=True)
            cls.config = config
            if not os.path.exists(config["data"]):
                download_data(gnn=True)

        def test_trainer(self):
            if not os.path.exists(self.config["checkpoint"]):
                download_checkpoint(gnn=True)
            train_model(self.config)

        def test_config(self):
            self.config = get_config(action='train', gnn=True)

    return TrainerTestEff


class TestTrainer(create_train_test_for_without_gnn()):
    'Test case for without gnn'


class TestTrainerEff(create_train_test_for_with_gnn()):
    'Test case for with gnn'


if __name__ == '__main__':

    unittest.main()