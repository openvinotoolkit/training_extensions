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

import os
import pathlib

from ote.tests.test_case import (create_export_test_case,
                                 create_test_case,
                                 skip_if_cuda_not_available)
from ote.utils.misc import run_through_shell


def create_image_classification_export_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    ExportTestCase = create_export_test_case('image_classification',
                                             **kwargs,
                                             metric_keys=['accuracy'],
                                             expected_outputs_dir=expected_outputs_dir)

    class ClassificationExportTestCase(ExportTestCase):
        def test_export_on_gpu(self):
            skip_if_cuda_not_available()
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
                    f'python3 export.py --openvino'
                    f' --load-weights snapshot.pth'
                    f' --save-model-to {export_dir}'
                )
                self.assertTrue(len(list(pathlib.Path(export_dir).rglob('*.onnx'))) > 0, 'Export to onnx failed')
                self.assertTrue(len(list(pathlib.Path(export_dir).rglob('*.bin'))) > 0, 'Export to openvino failed')

    return ClassificationExportTestCase


def create_image_classification_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    TrainTestCase = create_test_case('image_classification',
                                     **kwargs,
                                     metric_keys=['accuracy'],
                                     expected_outputs_dir=expected_outputs_dir,
                                     batch_size=2)
    class ClassificationTrainTestCase(TrainTestCase):
        def do_finetuning(self, on_gpu):
            self.total_epochs = 5
            log_file = os.path.join(self.output_folder, 'test_finetuning.log')
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python3 train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {os.path.join(self.img_root, "train")}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {os.path.join(self.img_root, "val")}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size {self.batch_size}'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')

            self.assertTrue(os.path.exists(os.path.join(self.output_folder, 'latest.pth')))
    return ClassificationTrainTestCase
