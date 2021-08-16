import unittest
import os
import torch
from torch.utils.data import DataLoader
from chest_xray_screening.utils.dataloader import RSNADataSet
from chest_xray_screening.utils.model import DenseNet121
from chest_xray_screening.inference import RSNAInference
from chest_xray_screening.utils.download_weights import download_checkpoint
from chest_xray_screening.utils.get_config import get_config


class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        export_config = get_config(action = 'test')
        cls.config = export_config
        if not os.path.isdir('model_weights'):
            download_checkpoint()
        image_path = '../../../data/chest_xray_screening/'
        dataset_test = RSNADataSet(
            cls.config['dummy_valid_list'],
            cls.config['dummy_labels'],
            image_path, transform = True)
        cls.data_loader_test = DataLoader(
            dataset=dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=False)

    def test_scores(self):
        self.model = DenseNet121(self.config['clscount'])
        self.class_names = self.config['class_names']
        target_metric = self.config['target_metric']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference = RSNAInference(
            self.model, self.data_loader_test,
            self.config['clscount'], self.config['checkpoint'],
            self.config['class_names'], device)
        metric = inference.test()
        self.assertGreaterEqual(metric, target_metric)


    def test_config(self):
        self.config = get_config(action = 'test')
        self.assertEqual(self.config['clscount'], 3)

    def test_config_eff(self):
        self.config = get_config(action = 'test', optimised = True)
        self.assertEqual(self.config['clscount'], 3)
        self.assertGreaterEqual(self.config['alpha'], 0)
        self.assertGreaterEqual(self.config['phi'], 0)
        self.assertLessEqual(self.config['alpha'], 2)
        self.assertLessEqual(self.config['phi'], 1)



if __name__ == '__main__':

    unittest.main()
