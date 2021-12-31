import unittest
import os
from torch.utils import data
from src.dataloader import Kaggle_BUS_Dataset
from src.solve import solver_inter2d
from src.utils.read_config import load_config
from src.utils.make_dirs import make_dirs

class TestInference(unittest.TestCase):
    net_config_path = "configs/net_config.json"
    net_config = load_config(net_config_path)
    test_config = net_config['test']
    infer_config = net_config['infer']
    data_config = net_config['data']
    gen_config = net_config['general']
    params_test = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 1,
                'drop_last': False}

    def test_paths_2d(self):
        self.stage0_path = self.data_config['stage0_data']
        self.realUS_path = self.data_config['realUS_data']
        self.checkpoint = os.path.join("temp_data","checkpoints", self.infer_config['exp_name'])
        if not os.path.exists(self.checkpoint):
            make_dirs()
        self.assertTrue(os.path.exists(self.stage0_path))
        self.assertTrue(os.path.exists(self.realUS_path))
        self.assertTrue(os.path.exists(self.checkpoint))

    def test_inference_2d(self):
        all_files = os.listdir(self.data_config['stage0_data'])
        test_ids = [temp for temp in all_files if temp[-6:-4] == "08"]
        self.assertTrue(test_ids)
        self.assertGreaterEqual(self.infer_config['dilation_factor'], 0.0)
        testing_set_2d = Kaggle_BUS_Dataset(test_ids, self.data_config['stage0_data'], self.data_config['realUS_data'])
        testing_gen = data.DataLoader(testing_set_2d, **self.params_test)
        self.inferpath = os.path.join("temp_data","infer_results", self.infer_config['exp_name'])
        solver_ins = solver_inter2d(self.infer_config, test_data=testing_gen,
                      test_flag=self.gen_config['test_flag'], restore=1, run_type='pytorch')
        solver_ins.test()
        pth_files_len = len(os.listdir(os.path.join(self.inferpath,'pytorch')))
        self.assertGreaterEqual(pth_files_len, 0)
        solver_onnx = solver_inter2d(
          self.infer_config, test_data=testing_gen,
          test_flag=self.gen_config['test_flag'],
          restore=1, run_type='onnx')
        solver_onnx.test()
        onnx_files_len = len(os.listdir(os.path.join(self.inferpath,'onnx')))
        self.assertGreaterEqual(onnx_files_len, 0)
        solver_ir = solver_inter2d(self.infer_config, test_data=testing_gen,
                        test_flag=self.gen_config['test_flag'], restore=1, run_type='ir')
        solver_ir.test()
        ir_files_len = len(os.listdir(os.path.join(self.inferpath,'ir')))
        self.assertGreaterEqual(ir_files_len, 0)

if __name__ == '__main__':
    unittest.main()
