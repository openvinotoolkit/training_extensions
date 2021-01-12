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
from ote.tests.utils import collect_ap, run_through_shell

from common.test_case import create_object_detection_export_test_case


kwargs = dict(
    problem_name='custom-object-detection',
    ann_file=os.path.dirname(__file__) + '/../../../data/airport/annotation_example_train.json',
    img_root=os.path.dirname(__file__) + '/../../../data/airport/train',
    alt_ssd_export=True
)


def create_custom_object_detection_export_test_case(model_name):
    class ExportTestCase(create_object_detection_export_test_case(model_name=model_name, **kwargs)):
        def do_evaluation(self, export_dir):
            metrics_path = os.path.join(export_dir, "metrics.yaml")
            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --load-weights {os.path.join(export_dir, "model.bin")}'
                f' --save-metrics-to {metrics_path}'
            )

            with open(metrics_path) as read_file:
                content = yaml.safe_load(read_file)

            for metric_key in self.metrics:
                value = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == metric_key][0]
                self.assertGreaterEqual(value, self.expected_outputs[metric_key] - self.test_export_thr)

        def do_export(self, export_dir, on_gpu):
            if not os.path.exists(export_dir):
                initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
                run_through_shell(
                    f'{initial_command}'
                    f'cd {os.path.dirname(self.template_file)};'
                    f'pip install -r requirements.txt;'
                    f'python export.py'
                    f' --load-weights snapshot.pth'
                    f' --save-model-to {export_dir}'
                )

    return ExportTestCase


class MobilenetV2_2S_SSD_256x256_ExportTestCase(
        create_custom_object_detection_export_test_case(
            model_name='mobilenet_v2-2s_ssd-256x256',
        )
):
    """ Test case for mobilenet_v2-2s_ssd-256x256 model export. """


class MobilenetV2_2S_SSD_384x384_ExportTestCase(
        create_custom_object_detection_export_test_case(
            model_name='mobilenet_v2-2s_ssd-384x384',
        )
):
    """ Test case for mobilenet_v2-2s_ssd-384x384 model export. """


class MobilenetV2_2S_SSD_512x512_ExportTestCase(
        create_custom_object_detection_export_test_case(
            model_name='mobilenet_v2-2s_ssd-512x512',
        )
):
    """ Test case for mobilenet_v2-2s_ssd-512x512 model export. """
