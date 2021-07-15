import unittest
import os
import json
import torch
from tools.utils.dataloader import RSNADataSet
from tools.utils.model import DenseNet121
from torch.utils.data import DataLoader
from tools.inference import RSNAInference


def get_config(optimised=False):
    path = os.path.dirname(os.path.realpath(__file__))
    with open(path+'/test_config.json','r') as f1:
        config_file = json.load(f1)

    if optimised:
        config = config_file['test_eff']
    else:
        config = config_file['test']

    return config


class InferenceTest(unittest.TestCase):
    config = get_config()
    checkpoint = config['checkpoint']
    class_count = config["clscount"]
    test_labels = config['dummy_test_labels']
    test_list = config['dummy_test_list']
    image_path = config["imgpath"]
    labels = config["labels"]
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
        target_metric = self.config['traget_metric']
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
