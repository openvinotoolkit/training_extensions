import unittest
import os
# import sys
from torch.utils import data
# sys.path.append("../src")
from src.dataloader import Kaggle_BUS_Dataset
from src.solve import solver
from src.utils.read_config import load_config
from src.utils.make_dirs import make_dirs

class TestTrain(unittest.TestCase):
  net_config_path = "configs/net_config.json"
  net_config = load_config(net_config_path)
  train_config = net_config['train']
  data_config = net_config['data']
  gen_config = net_config['general']
  params = {'batch_size': train_config['batch_size'],
          'shuffle': True,
          'num_workers': train_config['num_workers'],
          'drop_last': True}
  if not os.path.exists('downloads'):
      make_dirs()

  def test_paths(self):
    self.stage0_path = self.data_config['stage0_data']
    self.realUS_path = self.data_config['realUS_data']
    self.assertTrue(os.path.exists(self.stage0_path))
    self.assertTrue(os.path.exists(self.realUS_path))

  def test_train(self):
    all_files = os.listdir(self.data_config['stage0_data'])
    self.assertTrue(all_files)
    training_set = Kaggle_BUS_Dataset(all_files, self.data_config['stage0_data'], self.data_config['realUS_data'], resize=True)
    training_gen = data.DataLoader(training_set, **self.params)
    solver_ins = solver(self.train_config, self.gen_config, training_gen,  restore=self.train_config['restore'])
    solver_ins.train()

if __name__ == '__main__':
  unittest.main()
