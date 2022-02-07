import unittest
import numpy as np
from torch.utils.data import DataLoader
import os
from mammogram_screening.train_utils.dataloader import Stage1Dataset,Stage2bDataset
from mammogram_screening.train_utils.get_config import get_config
from mammogram_screening.train_utils.downloader import download_data#, prepare_data
from mammogram_screening.stage1.inference_mass_localization import InferenceStage1
from mammogram_screening.stage2.inference_stage2 import InferenceStage2


def create_inference_test_for_stage1():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='inference', stage='stage1')
            batch_sz = config['batch_size']
            num_workers = config['num_workers']
            cls.device = 'cuda'
            val_data_pth = config['val_data_path']
            cls.model_path = config['checkpoint']
            cls.onnx_model_path = config['onnx_checkpoint']

            if os.path.exists(val_data_pth):
                download_data()
                # prepare_data()

            x_tst = np.load(val_data_pth, allow_pickle=True)
            tst_data = Stage1Dataset(x_tst, transform=None)
            cls.tst_loader = DataLoader(tst_data, batch_size=batch_sz, shuffle=False, num_workers=num_workers)

        def test_pytorch_inference(self):
            inference = InferenceStage1(dataloader_test=self.tst_loader, checkpoint=self.model_path, device=self.device)
            model = inference.load_model(type='pytorch')
            mean_dice = inference.inference(model, runtype='pytorch')
            self.assertGreaterEqual(mean_dice, 0)
        
        def test_onnx_inference(self):
            inference = InferenceStage1(dataloader_test=self.tst_loader, checkpoint=self.onnx_model_path, device=self.device)
            model = inference.load_model(type='onnx')
            mean_dice = inference.inference(model, runtype='onnx')
            self.assertGreaterEqual(mean_dice, -1)

        def test_ir_inference(self):
            inference = InferenceStage1(dataloader_test=self.tst_loader, checkpoint=self.onnx_model_path, device='cpu')
            model = inference.load_model(type='cpu')
            mean_dice = inference.inference(model, runtype='cpu')
            self.assertGreaterEqual(mean_dice, -1)
    return InferenceTest



def create_inference_test_for_stage2():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='inference', stage='stage2')
            num_workers = cls.config['num_workers']
            cls.device = 'cuda'
            test_bags_path = cls.config['test_bags_path']
            if not os.path.exists(test_bags_path):
                download_data()
                # prepare_data()

            x_tst = np.load(test_bags_path, allow_pickle=True)
            tst_data = Stage2bDataset(x_tst, transform=None)
            cls.tst_loader = DataLoader(tst_data, batch_size=1, shuffle=False, num_workers=num_workers)

        def test_pytorch_inference(self):
            inference = InferenceStage2(test_loader=self.tst_loader, checkpoint=self.config['checkpoint'], device=self.device)
            model = inference.load_model(type='pytorch')
            test_acc, auc = inference.inference(model, run_type='pytorch', out_nm=self.config['out_pred_np'])

            self.assertGreaterEqual(test_acc, 0)
            self.assertGreaterEqual(auc, 0)
        
        def test_onnx_inference(self):
            inference = InferenceStage2(test_loader=self.tst_loader, checkpoint=self.config['onnx_checkpoint'], device=self.device)
            model = inference.load_model(type='onnx')
            test_acc, auc = inference.inference(model, run_type='onnx', out_nm=self.config['out_pred_np'])

            self.assertGreaterEqual(test_acc, -1)
            self.assertGreaterEqual(auc, 0)

        def test_ir_inference(self):
            inference = InferenceStage2(test_loader=self.tst_loader, checkpoint=self.config['onnx_checkpoint'], device='cpu')
            model = inference.load_model(type='ir')
            test_acc, auc = inference.inference(model, run_type='ir', out_nm=self.config['out_pred_np'])
            self.assertGreaterEqual(test_acc, -1)
            self.assertGreaterEqual(auc, 0)
    return InferenceTest

class TestInferenceStage1(create_inference_test_for_stage1()):
    'Test case for Stage1'

class TestInferenceStage2(create_inference_test_for_stage2()):
    'Test case for Stage2'



if __name__ == '__main__':

    unittest.main()
