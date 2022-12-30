import unittest
import os
import torch
from torch.utils.data import DataLoader
from chest_xray_screening.utils.dataloader import RSNADataSet
from chest_xray_screening.utils.model import DenseNet121, DenseNet121Eff
from chest_xray_screening.inference import RSNAInference
from chest_xray_screening.utils.download_weights import download_checkpoint, download_data
from chest_xray_screening.utils.get_config import get_config
from chest_xray_screening.utils.exporter import Exporter

def create_inference_test_for_densenet121():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            export_config = get_config(action = 'test')
            cls.config = export_config
            if not os.path.exists(cls.config['checkpoint']):
                download_checkpoint()
            if os.path.exists(export_config["default_image_path"]):
                cls.image_path = export_config["default_image_path"]
            else:
                if not os.path.exists(export_config["image_path"]):
                    download_data()
                cls.image_path = export_config["image_path"]

            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dataset_test = RSNADataSet(
                cls.config['dummy_valid_list'],
                cls.config['dummy_labels'],
                cls.image_path, transform = True)
            cls.data_loader_test = DataLoader(
                dataset=dataset_test,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=False)
            cls.model = DenseNet121(cls.config['clscount'])
            cls.inference = RSNAInference(
                cls.model, cls.data_loader_test,
                cls.config['clscount'], cls.config['checkpoint'],
                cls.config['class_names'], cls.device)
            cls.exporter = Exporter(cls.config, optimised=False)


        def test_pytorch_inference(self):
            metric = self.inference.validate_models(run_type='pytorch')
            self.assertGreaterEqual(metric, self.config['target_metric'])

        def test_onnx_inference(self):
            model_dir = os.path.split(self.config['checkpoint'])[0]
            onnx_checkpoint = os.path.join(model_dir, self.config.get('model_name_onnx'))
            if not os.path.exists(onnx_checkpoint):
                self.exporter.export_model_onnx()
            sample_image_name = self.config['dummy_valid_list'][0]
            sample_image_path = os.path.join(self.image_path, sample_image_name)
            self.inference.test_onnx(sample_image_path, onnx_checkpoint)
            metric = self.inference.validate_models(run_type='onnx', onnx_checkpoint=onnx_checkpoint)
            self.assertGreaterEqual(metric, self.config['target_metric'])

        def test_ir_inference(self):
            model_dir = os.path.split(self.config['checkpoint'])[0]
            onnx_checkpoint = os.path.join(model_dir, self.config.get('model_name_onnx'))
            if not os.path.exists(onnx_checkpoint):
                self.exporter.export_model_onnx()
            metric = self.inference.validate_models(run_type='openvino', onnx_checkpoint=onnx_checkpoint)
            self.assertGreaterEqual(metric, self.config['target_metric'])

        def test_config(self):
            self.config = get_config(action = 'test')
            self.assertEqual(self.config['clscount'], 3)

    return InferenceTest

def create_inference_test_for_densenet121eff():
    class InferenceTestEff(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            export_config = get_config(action = 'test', optimised = True)
            cls.config = export_config
            if not os.path.exists(cls.config['checkpoint']):
                download_checkpoint()
            if os.path.exists(export_config["default_image_path"]):
                cls.image_path = export_config["default_image_path"]
            else:
                if not os.path.exists(export_config["image_path"]):
                    download_data()
                cls.image_path = export_config["image_path"]
            dataset_test = RSNADataSet(
                cls.config['dummy_valid_list'],
                cls.config['dummy_labels'],
                cls.image_path, transform = True)
            cls.data_loader_test = DataLoader(
                dataset=dataset_test,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=False)

            alpha =  cls.config['alpha'] ** cls.config['phi']
            beta = cls.config['beta'] ** cls.config['phi']
            cls.model = DenseNet121Eff(alpha, beta, cls.config['class_count'])
            cls.class_names = cls.config['class_names']
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls.inference = RSNAInference(
                cls.model, cls.data_loader_test,
                cls.config['class_count'], cls.config['checkpoint'],
                cls.config['class_names'], cls.device)
            cls.exporter = Exporter(cls.config, optimised=True)

        def test_pytorch_inference(self):
            target_metric = self.config['target_metric']
            metric = self.inference.validate_models(run_type='pytorch')
            self.assertGreaterEqual(metric, target_metric)

        def test_onnx_inference(self):
            model_dir = os.path.split(self.config['checkpoint'])[0]
            onnx_checkpoint = os.path.join(model_dir, self.config.get('model_name_onnx'))
            if not os.path.exists(onnx_checkpoint):
                self.exporter.export_model_onnx()
            metric = self.inference.validate_models(run_type='onnx', onnx_checkpoint=onnx_checkpoint)
            self.assertGreaterEqual(metric, self.config['target_metric'])

        def test_ir_inference(self):
            model_dir = os.path.split(self.config['checkpoint'])[0]
            onnx_checkpoint = os.path.join(model_dir, self.config.get('model_name_onnx'))
            if not os.path.exists(onnx_checkpoint):
                self.exporter.export_model_onnx()
            metric = self.inference.validate_models(run_type='openvino', onnx_checkpoint=onnx_checkpoint)
            self.assertGreaterEqual(metric, self.config['target_metric'])

        def test_config_eff(self):
            self.assertEqual(self.config['class_count'], 3)
            self.assertGreaterEqual(self.config['alpha'], 0)
            self.assertGreaterEqual(self.config['phi'], -1)
            self.assertLessEqual(self.config['alpha'], 2)
            self.assertLessEqual(self.config['phi'], 1)

    return InferenceTestEff

class TestInference(create_inference_test_for_densenet121()):
    'Test case for DenseNet121'

class TestInferenceEff(create_inference_test_for_densenet121eff()):
    'Test case for DenseNet121Eff'

if __name__ == '__main__':

    unittest.main()
