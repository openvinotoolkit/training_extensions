# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import os
import pathlib
import unittest

import torch
import yaml

from ote.tests.test_case import (create_export_test_case,
                                 create_nncf_test_case,
                                 create_test_case)
from ote.tests.utils import run_through_shell


def create_image_classification_export_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    ExportTestCase = create_export_test_case('image_classification',
                                             **kwargs,
                                             metric_keys=['accuracy'],
                                             expected_outputs_dir=expected_outputs_dir)

    class ClassificationExportTestCase(ExportTestCase):
            @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
            def test_export_on_gpu(self):
                export_dir = os.path.join(self.output_folder, 'gpu_export')
                self.do_export(export_dir, on_gpu=True)

            def test_export_on_cpu(self):
                export_dir = os.path.join(self.output_folder, 'cpu_export')
                self.do_export(export_dir, on_gpu=False)

            def do_export(self, export_dir, on_gpu):
                if not os.path.exists(export_dir):
                    initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
                    run_through_shell(
                        f'{initial_command}'
                        f'cd {os.path.dirname(self.template_file)};'
                        f'pip install -r requirements.txt;'
                        f'python export.py --openvino'
                        f' --load-weights snapshot.pth'
                        f' --save-model-to {export_dir}'
                    )
                    self.assertTrue(len(list(pathlib.Path(export_dir).rglob('*.onnx'))) > 0, 'Export to onnx failed')
                    self.assertTrue(len(list(pathlib.Path(export_dir).rglob('*.bin'))) > 0, 'Export to openvino failed')

    return ClassificationExportTestCase

def create_image_classification_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    return create_test_case('image_classification',
                            **kwargs,
                            metric_keys=['accuracy'],
                            expected_outputs_dir=expected_outputs_dir)
