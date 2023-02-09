import os
import numpy as np
import unittest
import torchvision
import sys
sys.path.append(r'/storage/adityak/federated/')
from src.utils.inference_utils import inference_model
from torch.utils.data import DataLoader
from src.utils.get_config import get_config
from src.utils.downloader import download_data

def create_inference_test_with_gnn():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='inference', gnn=True)

            if not os.path.exists(cls.config['data']):
                download_data(gnn=True)

        def test_pytorch_inference(self):

            config = get_config(action='inference', gnn=True)
            inference_model(config,'pytorch')
 
        def test_onnx_inference(self):

            config = get_config(action='inference', gnn=True)
            inference_model(config,'onnx')

        def test_ir_inference(self):

            config = get_config(action='inference', gnn=True)
            inference_model(config,'ir')

    return InferenceTest

def create_inference_test_without_gnn():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='inference', gnn=False)

            if not os.path.exists(cls.config['data']):
                download_data(gnn=False)

        def test_pytorch_inference(self):

            config = get_config(action='inference', gnn=False)
            inference_model(config,'pytorch')
 
        def test_onnx_inference(self):

            config = get_config(action='inference', gnn=False)
            inference_model(config,'onnx')

        def test_ir_inference(self):

            config = get_config(action='inference', gnn=False)
            inference_model(config,'ir')

    return InferenceTest


class TestTrainer(create_inference_test_without_gnn()):
    'Test case for without gnn'

class TestTrainerEff(create_inference_test_with_gnn()):
    'Test case for with gnn'

if __name__ == '__main__':

    unittest.main()