import unittest
import os
from torch.utils import data
from breast_ultrasound_simulation.stage1.src.dataloader import BUS_Dataset
from breast_ultrasound_simulation.stage1.src.solve import solver_inter2d
from breast_ultrasound_simulation.stage1.src.utils.read_config import load_config


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
        self.stage0_path = self.infer_config['infer_stage0_data']
        self.realUS_path = self.infer_config['infer_realUS_data']
        self.assertTrue(os.path.exists(self.stage0_path))
        self.assertTrue(os.path.exists(self.realUS_path))

    def test_inference_2d(self):
        all_files = os.listdir(self.infer_config['infer_stage0_data'])
        self.assertTrue(all_files)
        self.assertGreaterEqual(self.infer_config['dilation_factor'], 0.0)
        testing_set_2d = BUS_Dataset(all_files,
                                     self.infer_config['infer_stage0_data'],
                                     self.infer_config['infer_realUS_data'])
        testing_gen = data.DataLoader(testing_set_2d, **self.params_test)
        self.inferpath = os.path.join("downloads", "checkpoints")
        solver_ins = solver_inter2d(self.infer_config, test_data=testing_gen,
                                    test_flag=self.gen_config['test_flag'], restore=1, run_type='pytorch')
        solver_ins.test()
        pth_files_exists = os.path.exists(os.path.join(self.inferpath, 'model.pt'))
        self.assertGreater(pth_files_exists, 0)
        solver_onnx = solver_inter2d(
            self.infer_config, test_data=testing_gen,
            test_flag=self.gen_config['test_flag'],
            restore=1, run_type='onnx')
        solver_onnx.test()
        onnx_files_exists = os.path.exists(os.path.join(self.inferpath, 'model.onnx'))
        self.assertGreater(onnx_files_exists, 0)
        solver_ir = solver_inter2d(
            self.infer_config, test_data=testing_gen,
            test_flag=self.gen_config['test_flag'], restore=1, run_type='ir')
        solver_ir.test()
        ir_files_exists = os.path.exists(os.path.join(self.inferpath, 'model.xml'))
        self.assertGreater(ir_files_exists, 0)


if __name__ == '__main__':
    unittest.main()
