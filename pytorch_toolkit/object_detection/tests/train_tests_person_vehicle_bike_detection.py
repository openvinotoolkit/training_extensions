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

import json
import os
import tempfile
import unittest

import yaml
from common.utils import replace_text_in_file, collect_ap, download_if_not_yet, run_through_shell


def test_case(model_name, snapshot_name):
    class Class(unittest.TestCase):

        def setUp(self):
            self.model_name = model_name

            self.data_folder = '../../data'
            self.work_dir = os.path.join('/tmp/', self.model_name)
            os.makedirs(self.work_dir, exist_ok=True)
            self.configuration_file = f'./person-vehicle-bike-detection/{self.model_name}/config.py'
            run_through_shell(f'cp {self.configuration_file} {self.work_dir}/')
            self.configuration_file = os.path.join(self.work_dir,
                                                   os.path.basename(self.configuration_file))
            self.ote_url = 'https://download.01.org/opencv/openvino_training_extensions'
            self.url = f'{self.ote_url}/models/object_detection/v2/{snapshot_name}'
            download_if_not_yet(self.work_dir, self.url)

            assert replace_text_in_file(self.configuration_file, 'samples_per_gpu=',
                                        'samples_per_gpu=1 ,#')
            assert replace_text_in_file(self.configuration_file, 'total_epochs = 20',
                                        'total_epochs = 25')
            assert replace_text_in_file(self.configuration_file, 'work_dir =',
                                        f'work_dir = "{os.path.join(self.work_dir, "outputs")}" #')
            assert replace_text_in_file(self.configuration_file, 'annotation_example_val.json',
                                        'annotation_example_train.json')
            assert replace_text_in_file(self.configuration_file, "data_root + 'val'",
                                        "data_root + 'train'")
            assert replace_text_in_file(self.configuration_file, 'resume_from = None',
                                        f'resume_from = "{os.path.join(self.work_dir, snapshot_name)}"')

        def test_fine_tuning(self):
            log_file = os.path.join(self.work_dir, 'test_fine_tuning.log')
            run_through_shell(
                f'../../external/mmdetection/tools/dist_train.sh {self.configuration_file} 1 2>&1 |'
                f' tee {log_file}')
            ap = collect_ap(log_file)
            self.assertEqual(len((ap)), 5)

        def test_quality_metrics(self):
            log_file = os.path.join(self.work_dir, 'test_quality_metrics.log')
            run_through_shell(
                f'python ../../external/mmdetection/tools/test.py '
                f'{self.configuration_file} '
                f'{os.path.join(self.work_dir, snapshot_name)} '
                f'--out res.pkl --eval bbox 2>&1 | tee {log_file}')
            ap = collect_ap(log_file)

            with open(f'tests/expected_outputs/person-vehicle-bike-detection/{self.model_name}.json') as read_file:
                content = json.load(read_file)

            self.assertEqual(content['map'], ap[0])

    return Class


class PersonVehicleBikeDetection2000TestCase(test_case('person-vehicle-bike-detection-2000',
                                                       'vehicle-person-bike-detection-2000-1.pth')):
    """ Test case for person-vehicle-bike-detection-2000 model. """


class PersonVehicleBikeDetection2001TestCase(test_case('person-vehicle-bike-detection-2001',
                                                       'vehicle-person-bike-detection-2001-1.pth')):
    """ Test case for person-vehicle-bike-detection-2001 model. """


class PersonVehicleBikeDetection2002TestCase(test_case('person-vehicle-bike-detection-2002',
                                                       'vehicle-person-bike-detection-2002-1.pth')):
    """ Test case for person-vehicle-bike-detection-2002 model. """


class PersonVehicleBikeDetection2000TestCaseOteApi(unittest.TestCase):

    @staticmethod
    def get_dependencies(template_file):
        output = {}
        with open(template_file) as read_file:
            content = yaml.load(read_file)
            for dependency in content['dependencies']:
                output[dependency['destination'].split('.')[0]] = dependency['source']
        return output

    def test_ok(self):
        self.model_name = 'person-vehicle-bike-detection-2000'

        self.template_file = f'./person-vehicle-bike-detection/{self.model_name}/template.yml'
        self.ann_file = '../../../../data/airport/annotation_example_train.json'
        self.img_root = ' ../../../../data/airport/train'
        self.ote_url = 'https://download.01.org/opencv/openvino_training_extensions'
        self.work_dir = tempfile.mkdtemp()
        self.dependencies = self.get_dependencies(self.template_file)

        download_if_not_yet(self.work_dir, self.dependencies['snapshot'])

        run_through_shell(
            f'cd {os.path.dirname(self.template_file)};'
            f'python {self.dependencies["eval"]}'
            f' --test_ann_files {self.ann_file}'
            f' --test_img_roots {self.img_root}'
            f' --save_metrics_to {os.path.join(self.work_dir, "metrics.yaml")}'
            f' --load_weights {os.path.join(self.work_dir, os.path.basename(self.dependencies["snapshot"]))}')

        with open(os.path.join(self.work_dir, "metrics.yaml")) as read_file:
            content = yaml.load(read_file)

        ap0 = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == 'ap'][0]

        run_through_shell(
            f'cd {os.path.dirname(self.template_file)};'
            f'python {self.dependencies["train"]}'
            f' --train_ann_files {self.ann_file}'
            f' --train_img_roots {self.img_root}'
            f' --val_ann_files {self.ann_file}'
            f' --val_img_roots {self.img_root}'
            f' --resume_from {os.path.join(self.work_dir, os.path.basename(self.dependencies["snapshot"]))}'
            f' --save_checkpoints_to {self.work_dir}'
            f' --gpu_num 1'
            f' --epochs 25')

        run_through_shell(
            f'cd {os.path.dirname(self.template_file)};'
            f'python {self.dependencies["eval"]}'
            f' --test_ann_files {self.ann_file}'
            f' --test_img_roots {self.img_root}'
            f' --save_metrics_to {os.path.join(self.work_dir, "metrics.yaml")}'
            f' --load_weights {os.path.join(self.work_dir, "latest.pth")}')

        with open(os.path.join(self.work_dir, "metrics.yaml")) as read_file:
            content = yaml.load(read_file)

        ap = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == 'ap'][0]
        assert ap > ap0 * 0.9
