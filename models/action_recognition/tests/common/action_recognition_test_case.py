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

from ote.tests.test_case import create_export_test_case, create_test_case
from ote.utils.misc import run_through_shell


def create_action_recognition_export_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    ExportTestCase = create_export_test_case('action_recognition',
                                             **kwargs,
                                             metric_keys=['accuracy'],
                                             expected_outputs_dir=expected_outputs_dir)
    return ExportTestCase


def create_action_recognition_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    TrainTestCase = create_test_case('action_recognition',
                                     **kwargs,
                                     metric_keys=['accuracy'],
                                     expected_outputs_dir=expected_outputs_dir,
                                     batch_size=2)
    class ActionRecognitionTrainTestCase(TrainTestCase):
        def do_finetuning(self, on_gpu):
            self.total_epochs = 5
            log_file = os.path.join(self.output_folder, 'test_finetuning.log')
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python3 train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size {self.batch_size}'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')

            self.assertTrue(os.path.exists(os.path.join(self.output_folder, 'latest.pth')))
    return ActionRecognitionTrainTestCase
