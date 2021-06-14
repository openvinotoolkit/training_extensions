import unittest
import os
import json
from utils.dataloader import RSNADataSet
from utils.model import DenseNet121

def get_config(optimised=False):
    path = os.path.dirname(os.path.realpath(__file__))
    with open(path+'/test_config.json','r') as f1:
        config_file = json.load(f1)
    
    if optimised:
        return config_file['test_eff']

    else:
        return config_file['test']


class InferenceTest(unittest.TestCase):
        
    def test_paths(self):
        self.config = get_config()
        self.image_path = self.config["imgpath"]
        self.np_path = self.config["npypath"]
        self.checkpoint = self.config["checkpoint"]
        self.assertTrue(os.path.exists(self.image_path))
        self.assertTrue(os.path.exists(self.checkpoint))
        self.assertTrue(os.path.exists(self.np_path+'test_list.npy'))
        self.assertTrue(os.path.exists(self.np_path+'test_labels.npy'))

    def test_config(self):
        self.config = get_config()
        self.class_count = self.config["clscount"]
        self.assertEqual(self.class_count,3)

    def test_config_eff(self):
        self.config = get_config(optimised=True)
        self.class_count = self.config["clscount"]
        self.assertEqual(self.class_count,3)
        self.assertGreaterEqual(self.config['alpha'],0)
        self.assertGreaterEqual(self.config['phi'],0)
        self.assertLessEqual(self.config['alpha'],2)
        self.assertLessEqual(self.config['phi'],1)



if __name__ == '__main__':

    unittest.main()

        




