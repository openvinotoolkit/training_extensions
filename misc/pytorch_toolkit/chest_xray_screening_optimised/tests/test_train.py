import unittest
import os
import json


def _get_config_():
    path = os.path.dirname(os.path.realpath(__file__))
    with open(path+'/test_config.json','r') as f1:
        config_file = json.load(f1)

    return config_file['train']


class TrainerTest(unittest.TestCase):
        
    def test_paths(self):
        self.config = _get_config_()
        self.image_path = self.config["imgpath"]
        self.np_path = self.config["npypath"]
        self.assertTrue(os.path.exists(self.image_path))
        self.assertTrue(os.path.exists(self.np_path+'train_list.npy'))
        self.assertTrue(os.path.exists(self.np_path+'train_labels.npy'))
        self.assertTrue(os.path.exists(self.np_path+'valid_list.npy'))
        self.assertTrue(os.path.exists(self.np_path+'valid_labels.npy'))
        self.assertTrue(os.path.exists(self.np_path+'test_list.npy'))
        self.assertTrue(os.path.exists(self.np_path+'test_labels.npy'))

    def test_config(self):
        self.config = _get_config_()
        self.learn_rate = self.config["lr"]
        self.class_count = self.config["clscount"]
        self.assertGreaterEqual(self.learn_rate,1e-8)
        self.assertEqual(self.class_count,3)
        self.assertGreaterEqual(self.config['alpha'],0)
        self.assertGreaterEqual(self.config['phi'],0)
        self.assertLessEqual(self.config['alpha'],2)
        self.assertLessEqual(self.config['phi'],1)

if __name__ == '__main__':

    unittest.main()

        




