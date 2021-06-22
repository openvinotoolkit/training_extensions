import unittest
import os
import json
from utils.dataloader import RSNADataSet
from utils.model import DenseNet121
from torch.utils.data import DataLoader


def get_config(optimised=False):
    path = os.path.dirname(os.path.realpath(__file__))
    with open(path+'/test_config.json','r') as f1:
        config_file = json.load(f1)
    
    if optimised:
        return config_file['test_eff']

    else:
        return config_file['test']


class InferenceTest(unittest.TestCase):
    config = get_config()
    checkpoint = config['checkpoint']
    class_count = config["clscount"]
    test_labels = config['dummy_test_labels']
    test_list = config['dummy_test_list']
    dataset_test = RSNADataSet(test_list, test_labels, image_path, transform=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False,  num_workers=4, pin_memory=False)

        
    def test_paths(self):
        self.config = get_config()
        self.image_path = self.config["imgpath"]
        self.np_path = self.config["npypath"]
        self.checkpoint = self.config["checkpoint"]
        self.assertTrue(os.path.exists(self.image_path))
        self.assertTrue(os.path.exists(self.checkpoint))
        self.assertTrue(os.path.exists(self.np_path+'test_list.npy'))
        self.assertTrue(os.path.exists(self.np_path+'test_labels.npy'))

    def test_scores(self):
        self.model = DenseNet121(self.class_count)
        self.class_names = self.config["class_names"]
        self.target_metric = self.config['traget_metric']
        self.inference = RSNAInference(self.model,self.data_loader_test,self.class_count,self.checkpoint,self.class_names,self.device)
        metric = self.inference.test()
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

        




