import unittest
import os
import json
import torch
from torch.utils.data import DataLoader
from google_drive_downloader import GoogleDriveDownloader as gdd
import sys
sys.path.append(os.path.abspath('../chest_xray_screening'))
sys.path.append(os.path.abspath('../utils'))
from dataloader import RSNADataSet
from model import DenseNet121
from inference import RSNAInference



def get_config(optimised=False):
    path = os.path.dirname(os.path.realpath(__file__))
    with open(path+'/test_config.json','r') as f1:
        config_file = json.load(f1)

    if optimised:
        config = config_file['test_eff']
    else:
        config = config_file['test']

    return config

def download_checkpoint():
    os.makedirs('model_weights')
    gdd.download_file_from_google_drive(file_id='1z4HuSVXyD59BHhw93j-BVbx6In1HZQn2',
                                    dest_path='model_weights/chest_xray_screening.pth.tar',
                                    unzip=False)
    gdd.download_file_from_google_drive(file_id='1HUmG-wKRoKYxBdwu0_LX1ascBRmA-z5e',
                                    dest_path='model_weights/chest_xray_screening_eff.pth.tar',
                                    unzip=False)


class InferenceTest(unittest.TestCase):
    config = get_config()
    checkpoint = config['checkpoint']
    if not os.path.isdir('model_weights'):
        download_checkpoint()
    class_count = config["clscount"]
    test_list = config['dummy_valid_list']
    image_path = '../../../../data/chest_xray_screening/'
    labels = config["dummy_labels"]
    dataset_test = RSNADataSet(test_list, labels, image_path, transform=True)
    data_loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False)

    def test_scores(self):
        self.model = DenseNet121(self.class_count)
        self.class_names = self.config["class_names"]
        target_metric = self.config["target_metric"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference = RSNAInference(
            self.model,self.data_loader_test,
            self.class_count,self.checkpoint,
            self.class_names,device)
        metric = inference.test()
        self.assertGreaterEqual(metric, target_metric)


    def test_config(self):
        self.config = get_config()
        self.assertEqual(self.class_count,3)

    def test_config_eff(self):
        self.config = get_config(optimised=True)
        self.assertEqual(self.class_count,3)
        self.assertGreaterEqual(self.config['alpha'],0)
        self.assertGreaterEqual(self.config['phi'],0)
        self.assertLessEqual(self.config['alpha'],2)
        self.assertLessEqual(self.config['phi'],1)



if __name__ == '__main__':

    unittest.main()
