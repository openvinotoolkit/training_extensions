import os
import numpy as np
import unittest
import torchvision
from src.utils.inference_utils import load_inference_model, validate_model
from torch.utils.data import DataLoader
from src.utils.get_config import get_config
from src.utils.dataloader import CustomDatasetPhase1, CustomDatasetPhase2
from src.utils.downloader import download_data


def create_inference_test_for_phase1():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='inference', phase=1)

            if not os.path.exists(cls.config['image_path']):
                download_data(phase=1)
                # prepare_data()

            # x_tst = np.load(cls.config['image_path'], allow_pickle=True)
            cls.images_transforms = torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])

            tst_data = CustomDatasetPhase1(cls.config['image_path'], transform_images=cls.images_transforms)
            cls.tst_loader = DataLoader(tst_data, batch_size=1, shuffle=False, num_workers=16)

        def test_pytorch_inference(self):
            config = get_config(action='inference', phase=1)

            model = load_inference_model(config=config, run_type='pytorch')

            # getting avg_ssim and avg_psnr
            ssim, psnr = validate_model(model=model, config=config, run_type='pytorch')

            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)

        def test_onnx_inference(self):

            config = get_config(action='inference', phase=1)
            model = load_inference_model(config=config, run_type='onnx')
            # getting avg_ssim and avg_psnr
            ssim, psnr = validate_model(model=model, config=config, run_type='onnx')

            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)

        def test_ir_inference(self):

            config = get_config(action='inference', phase=1)
            model = load_inference_model(config=config, run_type='ir')

            # getting avg_ssim and avg_psnr
            ssim, psnr = validate_model(model=model, config=config, run_type='ir')

            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)

    return InferenceTest

def create_inference_test_for_phase2():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='inference', phase=2)

            if not os.path.exists(cls.config['image_path']):
                download_data(phase=2)

            tst_data = CustomDatasetPhase2(path_to_latent=cls.config['path_to_latent'], path_to_gdtruth=cls.config['path_to_gdtruth'])
            cls.tst_loader = DataLoader(tst_data, batch_size=1, shuffle=False, num_workers=16)

        def test_pytorch_inference(self):
            config = get_config(action='inference', phase=2)

            model = load_inference_model(config=config, run_type='pytorch')

            # getting avg_ssim and avg_psnr
            ssim, psnr = validate_model(model=model, config=config, run_type='pytorch')
            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)

        def test_onnx_inference(self):

            config = get_config(action='inference', phase=2)
            model = load_inference_model(config=config, run_type='onnx')
            # getting avg_ssim and avg_psnr
            ssim, psnr = validate_model(model=model, config=config, run_type='onnx')
            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)

        def test_ir_inference(self):
            config = get_config(action='inference', phase=2)

            model = load_inference_model(config=config, run_type='ir')

            # getting avg_ssim and avg_psnr
            ssim, psnr = validate_model(
                model=model, config=config, run_type='ir')
            self.assertGreater(ssim, 0.8)
            self.assertGreater(psnr, 25)

    return InferenceTest


class TestTrainer(create_inference_test_for_phase1()):
    'Test case for phase1'


class TestTrainerEff(create_inference_test_for_phase2()):
    'Test case for phase2'


if __name__ == '__main__':

    unittest.main()
