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

import yaml
from ote.tests.utils import collect_ap
from ote.utils.misc import run_through_shell

from common.test_case import create_object_detection_test_case


kwargs = dict(
    problem_name='custom-object-detection',
    ann_file=os.path.dirname(__file__) + '/../../../data/airport/annotation_example_train.json',
    img_root=os.path.dirname(__file__) + '/../../../data/airport/train'
)

def create_custom_object_detection_test_case(model_name):
    class TestCase(create_object_detection_test_case(model_name=model_name, **kwargs)):
        classes = 'vehicle,person,non-vehicle'

        def do_evaluation(self, on_gpu):
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            metrics_path = os.path.join(self.output_folder, "metrics.yaml")
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to {metrics_path}'
                f' --load-weights snapshot.pth'
                f' --classes {self.classes}'
            )

            with open(metrics_path) as read_file:
                content = yaml.safe_load(read_file)

            for metric_key in self.metrics:
                value = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == metric_key][0]
                self.assertGreaterEqual(value, self.expected_outputs[metric_key])

        def do_finetuning(self, on_gpu):
            log_file = os.path.join(self.output_folder, 'test_finetuning.log')
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --resume-from snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' --epochs 1'
                f' --classes {self.classes}'
                f' | tee {log_file}')

            self.assertTrue(os.path.exists(os.path.join(self.output_folder, 'latest.pth')))

    return TestCase


class MobilenetV2_2S_SSD_256x256_TestCase(
        create_custom_object_detection_test_case(
            model_name='mobilenet_v2-2s_ssd-256x256',
        )
):
    """ Test case for mobilenet_v2-2s_ssd-256x256 model. """


class MobilenetV2_2S_SSD_384x384_TestCase(
        create_custom_object_detection_test_case(
            model_name='mobilenet_v2-2s_ssd-384x384',
        )
):
    """ Test case for mobilenet_v2-2s_ssd-384x384 model. """


class MobilenetV2_2S_SSD_512x512_TestCase(
        create_custom_object_detection_test_case(
            model_name='mobilenet_v2-2s_ssd-512x512',
        )
):
    """ Test case for mobilenet_v2-2s_ssd-512x512 model. """
