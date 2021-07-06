import unittest
import os
import argparse
import sys
from torch.utils import data
sys.path.append("../src")
from dataloader import IVUS_Dataset
from src.solve import Solver_inter2d 
# from dataloader import IVUS3D_Dataset
# from src.solve import Solver_inter3d

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="testing", help='experiment name')
    parser.add_argument('--model_name', default="model.pt", help='model that needs to be loaded')
    parser.add_argument(
        '--stage0_data',
        default='../data/stage0',
        help='path to stage 0 output')
    parser.add_argument(
        '--realUS_data',
        default='../data/real_images',
        help='path to real ultrasound images')

    parser.add_argument(
        '--dilation_factor',
        type=float,
        default=0.0,
        help='controls the resolution only for IVUS2D and IVUS3D')
    parser.add_argument(
        '--infer_data',
        default='IVUS2D',
        help='select from IVUS2D, IVUS3D, BUS')
    parser.add_argument('--test', type=int, default=1)
    args = parser.parse_args()
    return args


class TestInference(unittest.TestCase):
    args = get_args()
    #testing_set_3d = IVUS_Dataset(test_ids, "../data/stage0_3d", "../data/real_images_3d")
    params_test = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 1,
                'drop_last': False}

    def test_paths_2d(self):
      self.stage0_path = self.args.stage0_data
      self.realUS_path = self.args.realUS_data
      self.checkpoint = os.path.join("../checkpoints", self.args.model_name)
      self.assertTrue(os.path.exists(self.stage0_path))
      self.assertTrue(os.path.exists(self.realUS_path))
      self.assertTrue(os.path.exists(self.checkpoint))

    # Tests for 3d data is optional.
    # Users can uncomment the lines to enable the tests. More details in ReadMe.
  
    # def test_paths_3d(self):
    #   self.stage0_path = "../data/stage0_3d"
    #   self.realUS_path = "../data/stage0_3d"
    #   self.assertTrue(os.path.exists(self.stage0_path))
    #   self.assertTrue(os.path.exists(self.realUS_path))
    

    def test_inference_2d(self):
      all_files = os.listdir(self.args.stage0_data)
      test_ids = [temp for temp in all_files if temp[6:8] == "09"]
      self.assertTrue(test_ids)
      self.assertEqual(self.args.test, 1)
      self.assertGreaterEqual(self.args.dilation_factor, 0)
      testing_set_2d = IVUS_Dataset(test_ids, self.args.stage0_data, self.args.realUS_data)
      testing_gen = data.DataLoader(testing_set_2d, **self.params_test)
      solver_ins = Solver_inter2d(self.args, test_data=testing_gen, restore=1)
      solver_ins.test()

    # def test_inference_3d(self):

    #   all_files = os.listdir("../data/stage0_3d")
    #   test_ids = [temp for temp in all_files if temp[1:5] == "09_0"]
    #   testing_set_3d = IVUS3D_Dataset(test_ids, "../data/stage0_3d", "../data/real_images_3d")
    #   testing_gen = data.DataLoader(testing_set_3d, **self.params_test)
    #   solver_ins = solver_inter3d(self.args, test_data=testing_gen, restore=1)
    #   solver_ins.test()

if __name__ == '__main__':
    unittest.main()
      
