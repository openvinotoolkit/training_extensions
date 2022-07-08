import os
import numpy as np
import unittest
import torch
from torch.utils.data import DataLoader
from utils import get_config
from utils.dataloader import CustomDatasetPhase1, CustomDatasetPhase2
from utils.downloader import download_data
from utils.train_utils import validate_model_phase1, validate_model_phase2


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
                tst_data, batch_size=1, shuffle=False, num_workers=16)

        def test_pytorch_inference(self):
            inference = CustomDatasetPhase1(
                dataloader_test=self.tst_loader, checkpoint=self.model_path,
                device=self.device)
            model = inference.load_model(run_type='pytorch')

            # getting avg_loss, avg_ssim and avg_psnr
            loss, ssim, psnr = validate_model_phase1(
                config=get_config(phase=1),
                test_dataloader=self.tst_loader,
                model=model, msecrit=torch.nn.MSELoss()
            )
            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)
            self.assertLesser(loss, 0.1)

        def test_onnx_inference(self):
            inference = CustomDatasetPhase1(
                dataloader_test=self.tst_loader, checkpoint=self.onnx_model_path,
                device=self.device)
            model = inference.load_model(run_type='onnx')

            # getting avg_loss, avg_ssim and avg_psnr
            loss, ssim, psnr = validate_model_phase1(
                config=get_config(phase=1),
                test_dataloader=self.tst_loader,
                model=model, msecrit=torch.nn.MSELoss()
            )
            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)
            self.assertLesser(loss, 0.1)

        def test_ir_inference(self):
            inference = CustomDatasetPhase1(
                dataloader_test=self.tst_loader, checkpoint=self.onnx_model_path,
                device='cpu')
            model = inference.load_model(run_type='ir')

            # getting avg_loss, avg_ssim and avg_psnr
            loss, ssim, psnr = validate_model_phase1(
                config=get_config(phase=1),
                test_dataloader=self.tst_loader,
                model=model, msecrit=torch.nn.MSELoss()
            )
            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)
            self.assertLesser(loss, 0.1)

    return InferenceTest


def create_inference_test_for_phase2():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='inference', phase=2)

            if os.path.exists(val_data_pth):
                download_data()
                # prepare_data()

            x_tst = np.load(val_data_pth, allow_pickle=True)
            tst_data = CustomDatasetPhase2()
            cls.tst_loader = DataLoader(
                tst_data, batch_size=1, shuffle=False, num_workers=16)

        def test_pytorch_inference(self):
            inference = CustomDatasetPhase2(
                dataloader_test=self.tst_loader, checkpoint=self.model_path,
                device=self.device)
            model = inference.load_model(run_type='pytorch')

            # getting avg_loss, avg_ssim and avg_psnr
            loss, ssim, psnr = validate_model_phase2(
                config=get_config(phase=1),
                test_dataloader=self.tst_loader,
                model=model, msecrit=torch.nn.MSELoss()
            )
            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)
            self.assertLesser(loss, 0.1)

        def test_onnx_inference(self):
            inference = CustomDatasetPhase1(
                dataloader_test=self.tst_loader, checkpoint=self.onnx_model_path,
                device=self.device)
            model = inference.load_model(run_type='onnx')

            # getting avg_loss, avg_ssim and avg_psnr
            loss, ssim, psnr = validate_model_phase2(
                config=get_config(phase=1),
                test_dataloader=self.tst_loader,
                model=model, msecrit=torch.nn.MSELoss()
            )
            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)
            self.assertLesser(loss, 0.1)

        def test_ir_inference(self):
            inference = CustomDatasetPhase1(
                dataloader_test=self.tst_loader, checkpoint=self.onnx_model_path,
                device='cpu')
            model = inference.load_model(run_type='ir')

            # getting avg_loss, avg_ssim and avg_psnr
            loss, ssim, psnr = validate_model_phase2(
                config=get_config(phase=1),
                test_dataloader=self.tst_loader,
                model=model, msecrit=torch.nn.MSELoss()
            )
            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)
            self.assertLesser(loss, 0.1)

    return InferenceTest
