import unittest
import os
from torch.utils.data import DataLoader
from chest_xray_screening.train import RSNATrainer
from chest_xray_screening.utils.dataloader import RSNADataSet
from chest_xray_screening.utils.model import DenseNet121
from chest_xray_screening.utils.download_weights import download_checkpoint
from chest_xray_screening.utils.get_config import get_config


class TrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = get_config(action = 'train')
        cls.config = config
        image_path = '../../../data/chest_xray_screening/'

        dataset_train = RSNADataSet(
            cls.config['dummy_train_list'],
            cls.config['dummy_labels'],
            image_path, transform=True)
        dataset_valid = RSNADataSet(
            cls.config['dummy_valid_list'],
            cls.config['dummy_labels'],
            image_path, transform=True)
        dataset_test = RSNADataSet(
            cls.config['dummy_test_list'],
            cls.config['dummy_labels'],
            image_path, transform=True)
        cls.data_loader_train = DataLoader(
            dataset=dataset_train,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            pin_memory=False)
        cls.data_loader_valid = DataLoader(
            dataset=dataset_valid,
            batch_size=2,
            shuffle=False,
            num_workers=4,
            pin_memory=False)
        cls.data_loader_test = DataLoader(
            dataset=dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=False)


    def test_config(self):
        self.assertGreaterEqual(self.config["lr"], 1e-8)
        self.assertEqual(self.config["clscount"], 3)

    def test_trainer(self):
        self.model = DenseNet121(self.config["clscount"])
        if not os.path.isdir('model_weights'):
            download_checkpoint()
        self.device = self.config["device"]
        self.trainer = RSNATrainer(
            self.model, self.data_loader_train,
            self.data_loader_valid, self.data_loader_test,
            self.config["clscount"], self.config["checkpoint"],
            self.device, self.config["class_names"], self.config["lr"])
        self.trainer.train(self.config["max_epoch"], self.config["savepath"])
        cur_train_loss = self.trainer.current_train_loss
        cur_valid_loss = self.trainer.current_valid_loss
        self.trainer.train(self.config["max_epoch"], self.config["savepath"])
        self.assertLessEqual(self.trainer.current_train_loss, cur_train_loss)
        self.assertLessEqual(self.trainer.current_valid_loss, cur_valid_loss)

    def test_config_eff(self):
        self.config = get_config(action = 'train', optimised = True)
        self.learn_rate = self.config["lr"]
        self.class_count = self.config["clscount"]
        self.assertGreaterEqual(self.learn_rate, 1e-8)
        self.assertEqual(self.class_count, 3)
        self.assertGreaterEqual(self.config['alpha'], 0)
        self.assertGreaterEqual(self.config['phi'], 0)
        self.assertLessEqual(self.config['alpha'], 2)
        self.assertLessEqual(self.config['phi'], 1)


if __name__ == '__main__':

    unittest.main()
