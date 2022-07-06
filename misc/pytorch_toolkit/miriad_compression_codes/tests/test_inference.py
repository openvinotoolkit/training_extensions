import os
import numpy as np
import unittest
from utils.dataloader import CustomDatasetPhase1, CustomDatasetPhase2
from utils import get_config
from torch.utils.data import DataLoader


def create_inference_test_for_phase1():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='inference', phase=1)

            if os.path.exists(val_data_pth):
                download_data()
                # prepare_data()

            x_tst = np.load(val_data_pth, allow_pickle=True)
            tst_data = CustomDatasetPhase1()
            cls.tst_loader = DataLoader(
                tst_data, batch_size=batch_sz, shuffle=False, num_workers=num_workers)

        def test_pytorch_inference(self):
            inference = CustomDatasetPhase1(
                dataloader_test=self.tst_loader, checkpoint=self.model_path, device=self.device)
            model = inference.load_model(run_type='pytorch')
            mean_dice = inference.inference(model, runtype='pytorch')
            self.assertGreaterEqual(mean_dice, 0)

        def test_onnx_inference(self):
            inference = CustomDatasetPhase1(
                dataloader_test=self.tst_loader,
                checkpoint=self.onnx_model_path,
                device=self.device)
            model = inference.load_model(run_type='onnx')
            mean_dice = inference.inference(model, runtype='onnx')
            self.assertGreaterEqual(mean_dice, -1)

        def test_ir_inference(self):
            inference = CustomDatasetPhase1(
                dataloader_test=self.tst_loader, checkpoint=self.onnx_model_path, device='cpu')
            model = inference.load_model(run_type='cpu')
            mean_dice = inference.inference(model, runtype='cpu')
            self.assertGreaterEqual(mean_dice, -1)
    return InferenceTest
