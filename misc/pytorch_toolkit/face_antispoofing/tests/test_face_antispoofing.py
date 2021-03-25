"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest
import torch
import sys

from utils import read_py_config, build_model, check_file_exist


class ExportError(Exception):
    pass


class TestONNXExport(unittest.TestCase):
    def setUp(self):
        config = read_py_config('./configs/config.py')
        self.config = config
        self.model = build_model(config, device='cpu', strict=True, mode='convert')
        self.img_size = tuple(map(int, config.resize.values()))

    def test_export(self):
        # input to inference model
        dummy_input = torch.rand(size=(1, 3, *(self.img_size)), device='cpu')
        self.model.eval()
        onnx_model_path = './mobilenetv3.onnx'
        torch.onnx.export(self.model, dummy_input, onnx_model_path, verbose=False)
        check_file_exist(onnx_model_path)

if __name__ == '__main__':
    unittest.main()
