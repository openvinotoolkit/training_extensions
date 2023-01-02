import unittest
import os
from torch.utils import data
from breast_ultrasound_simulation.stage1.src.dataloader import BUS_Dataset
from breast_ultrasound_simulation.stage1.src.solve import solver
from breast_ultrasound_simulation.stage1.src.utils.read_config import load_config
from breast_ultrasound_simulation.stage1.src.utils.download_mdl_tdataset import download_data
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

    def test_train(self):

        all_files = os.listdir(self.data_config['stage0_data'])
        self.assertTrue(all_files)
        training_set = BUS_Dataset(all_files, self.data_config['stage0_data'],
                          self.data_config['realUS_data'], resize=True)

        training_gen = data.DataLoader(training_set, **self.params)

        val_all_files = os.listdir(self.data_config['val_stage0_data'])
        self.assertTrue(val_all_files)
        validation_set = BUS_Dataset(all_files, self.data_config['val_stage0_data'],
                          self.data_config['val_realUS_data'], resize=True)

        testing_gen = data.DataLoader(validation_set, **self.params)

        solver_ins = solver(self.train_config, training_gen, testing_gen, restore=self.train_config['restore'])
        loss_epoch = solver_ins.train(num_epochs_to_run=self.train_config['epochs'])
        self.assertLessEqual(loss_epoch[-1], loss_epoch[0])

    def test_paths(self):
        self.stage0_path = self.data_config['stage0_data']
        if not os.path.exists(self.stage0_path):
            download_data()
        self.realUS_path = self.data_config['realUS_data']
        self.assertTrue(os.path.exists(self.stage0_path))
        self.assertTrue(os.path.exists(self.realUS_path))


if __name__ == '__main__':
    unittest.main()
