import unittest
import os
from src.utils.train_stage1 import train_network
from src.utils.train_stage2 import lungpatch_classifier
from src.utils.models import SUMNet, LeNet
from src.utils.downloader import download_checkpoint, download_data
from src.utils.get_config import get_config
from src.utils.utils import create_dummy_json_file

def create_train_test_for_stage1():
    class TrainerTestStage1(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train', stage=1)
            cls.config = config
            if not os.path.exists(config["datapath"]):
                download_data()
            if not os.path.exists(config["json_path"]):
                create_dummy_json_file(config["json_path"], stage=1)

        def test_trainer(self):
            self.model = SUMNet(in_ch=1,out_ch=2)
            loss_list= train_network(self.config)
            self.assertLessEqual(loss_list[2], loss_list[0])

        def test_config(self):
            self.config = get_config(action='train', stage=1)
            self.assertGreaterEqual(self.config["lrate"], 1e-8)

    return TrainerTestStage1

def create_train_test_for_stage2():
    class TrainerTestStage2(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train', stage=2)
            cls.config = config
            if not os.path.exists(config["imgpath"]):
                download_data()
            if not os.path.exists(config["jsonpath"]):
                create_dummy_json_file(config["jsonpath"], stage=2)


        def test_trainer(self):
            self.model = LeNet()
            loss_list = lungpatch_classifier(self.config)
            self.assertLessEqual(loss_list[1], loss_list[0])

        def test_config(self):
            self.config = get_config(action='train', stage=2)
            self.assertGreaterEqual(self.config["lrate"], 1e-8)

    return TrainerTestStage2


class TestTrainerStage1(create_train_test_for_stage1()):
    'Test case for stage1'

class TestTrainerStage2(create_train_test_for_stage2()):
    'Test case for stage2'


if __name__ == '__main__':

    unittest.main()
