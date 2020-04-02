#!/usr/bin/env python3
#
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

""" This module contains unit tests. """

import os
import tempfile
import unittest
import json


def replace_text_in_file(path, replace_what, replace_by):
    with open(path) as read_file:
        content = '\n'.join([line.rstrip() for line in read_file.readlines()])
        if content.find(replace_what) == -1:
            return False
        content = content.replace(replace_what, replace_by)
    with open(path, 'w') as write_file:
        write_file.write(content)
    return True


def collect_ap(path):
    ap = []
    beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file.readlines()]
        for line in content:
            if line.startswith(beginning):
                ap.append(float(line.replace(beginning, '')))
    return ap


def face_detection_test_case(model_name):
    class Class(unittest.TestCase):

        def setUp(self):
            self.model_name = model_name

            self.data_folder = '../../data'
            self.work_dir = tempfile.mkdtemp()
            self.configuration_file = f'./configs/{self.model_name}.py'
            os.system(f'cp {self.configuration_file} {self.work_dir}/')
            self.configuration_file = os.path.join(self.work_dir, os.path.basename(self.configuration_file))
            self.ote_url = 'https://download.01.org/opencv/openvino_training_extensions'
            self.url = f'{self.ote_url}/models/object_detection/{self.model_name}.pth'
            os.system(f'wget {self.url} -P {self.work_dir}')

            assert replace_text_in_file(self.configuration_file, 'imgs_per_gpu=', 'imgs_per_gpu=2 ,#')
            assert replace_text_in_file(self.configuration_file, 'total_epochs = 70', 'total_epochs = 75')
            assert replace_text_in_file(self.configuration_file, 'data/WIDERFace', '../../data/airport')
            assert replace_text_in_file(self.configuration_file, 'work_dir =',
                                        f'work_dir = "{os.path.join(self.work_dir, "outputs")}" #')
            assert replace_text_in_file(self.configuration_file, 'train.json', 'annotation_faces_train.json')
            assert replace_text_in_file(self.configuration_file, 'val.json', 'annotation_faces_train.json')
            assert replace_text_in_file(self.configuration_file, 'resume_from = None',
                                        f'resume_from = "{os.path.join(self.work_dir, self.model_name)}.pth"')

            os.system(f'cat {self.configuration_file}')

        def test_fine_tuning(self):
            log_file = os.path.join(self.work_dir, 'test_fine_tuning.log')
            os.system(
                f'../../external/mmdetection/tools/dist_train.sh {self.configuration_file} 1 --validate 2>&1 |'
                f' tee {log_file}')
            ap = collect_ap(log_file)
            self.assertEqual(len((ap)), 5)
            self.assertLess(ap[0], ap[-1])

        def test_quality_metrics(self):
            log_file = os.path.join(self.work_dir, 'test_quality_metrics.log')
            os.system(
                f'python ../../external/mmdetection/tools/test.py '
                f'{self.configuration_file} '
                f'{os.path.join(self.work_dir, self.model_name + ".pth")} '
                f'--out res.pkl --eval bbox 2>&1 | tee {log_file}')
            ap = collect_ap(log_file)

            with open(f'tests/expected_outputs/{self.model_name}.json') as read_file:
                content = json.load(read_file)

            self.assertEqual(content['map'], ap[0])

    return Class


class face_detection_0100_test_case(face_detection_test_case('face-detection-0100')):
    pass


class face_detection_0102_test_case(face_detection_test_case('face-detection-0102')):
    pass


class face_detection_0104_test_case(face_detection_test_case('face-detection-0104')):
    pass


class face_detection_0105_test_case(face_detection_test_case('face-detection-0105')):
    pass


class face_detection_0106_test_case(face_detection_test_case('face-detection-0106')):
    pass


if __name__ == '__main__':
    unittest.main()
