import unittest
import os
import argparse
import sys
from torch.utils import data
sys.path.append("../src")

from dataloader import IVUS_Dataset, IVUS3D_Dataset
from solve import solver

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', default="testing", help='experiment name')
	parser.add_argument('--model_name', help='model that needs to be loaded')
	parser.add_argument(
	    '--stage0_data',
	    default='../data/stage0',
	    help='path to stage 0 output')
	parser.add_argument(
	    '--realUS_data',
	    default='../data/real_images',
	    help='path to real ultrasound images')
	parser.add_argument('--batch_size', type=int, default=2)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--lr', type=float, default=0.0002)
	parser.add_argument(
	    '--log_step',
	    type=int,
	    default=2,
	    help="Logging frequency, set -1 for no logging")
	parser.add_argument('--vis_step', type=int, default=1,
			    help="Tensorboard visualization frequency")
	parser.add_argument(
	    '--restore',
	    type=int,
	    default=0,
	    help="restore a model for continuing training")
	parser.add_argument('--test', type=int, default=0)
	parser.add_argument('--beta_reco', type=float, default=1)
	parser.add_argument('--beta_adv', type=float, default=0.02)

	args = parser.parse_args()
	return args

class TestTrain(unittest.TestCase):
  args = get_args()
  params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'drop_last': True}

  def test_paths(self):
    self.stage0_path = self.args.stage0_data
    self.realUS_path = self.args.realUS_data
    self.assertTrue(os.path.exists(self.stage0_path))
    self.assertTrue(os.path.exists(self.realUS_path))

  def test_train(self):
    all_files = os.listdir(self.args.stage0_data)
    self.assertTrue(all_files)
    training_set = IVUS_Dataset(all_files, self.args.stage0_data, self.args.realUS_data, resize=True)
    training_gen = data.DataLoader(training_set, **self.params)
    solver_ins = solver(self.args, training_gen,  restore=self.args.restore)
    solver_ins.train()

if __name__ == '__main__':
  unittest.main()
