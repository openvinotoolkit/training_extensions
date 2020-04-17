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
import unittest

from common.utils import replace_text_in_file, collect_ap, download_if_not_yet


class PersonVehicleBikeDetectionCrossroad1016TestCase(unittest.TestCase):

    def setUp(self):
        self.model_name = 'person-vehicle-bike-detection-crossroad-1016'
        self.snapshot_name = 'person_vehicle_bike_sd512_mb2_clustered_epoch_21.pth'

        self.data_folder = '../../data'
        self.work_dir = os.path.join('/tmp', self.model_name)
        os.makedirs(self.work_dir, exist_ok=True)
        self.configuration_file = f'./configs/{self.model_name}.py'
        os.system(f'cp {self.configuration_file} {self.work_dir}/')
        self.configuration_file = os.path.join(self.work_dir,
                                               os.path.basename(self.configuration_file))
        self.ote_url = 'https://download.01.org/opencv/openvino_training_extensions'
        self.url = f'{self.ote_url}/models/object_detection/{self.snapshot_name}'
        download_if_not_yet(self.work_dir, self.url)

        assert replace_text_in_file(self.configuration_file, 'total_epochs =',
                                    'total_epochs = 25#')
        assert replace_text_in_file(self.configuration_file, 'work_dir =',
                                    f'work_dir = "{os.path.join(self.work_dir, "outputs")}" #')
        assert replace_text_in_file(self.configuration_file, 'load_from = None',
                                    f'load_from = "{os.path.join(self.work_dir, self.snapshot_name)}"')
        assert replace_text_in_file(self.configuration_file, 'annotation_example_train.json',
                                    'annotation_example_val.json')
        assert replace_text_in_file(self.configuration_file, '/train', '/val')

    def test_fine_tuning(self):
        log_file = os.path.join(self.work_dir, 'test_fine_tuning.log')

        os.system(
            f'../../external/mmdetection/tools/dist_train.sh {self.configuration_file} 1 --validate 2>&1 |'
            f' tee {log_file}')
        ap = collect_ap(log_file)
        self.assertEqual(len((ap)), 25)
        self.assertLess(ap[0], ap[-1])

    def test_quality_metrics(self):
        log_file = os.path.join(self.work_dir, 'test_quality_metrics.log')
        os.system(
            f'python ../../external/mmdetection/tools/test.py '
            f'{self.configuration_file} '
            f'{os.path.join(self.work_dir, self.snapshot_name)} '
            f'--out res.pkl --eval bbox 2>&1 | tee {log_file}')
        ap = collect_ap(log_file)

        with open(f'tests/expected_outputs/{self.model_name}.json') as read_file:
            content = json.load(read_file)

        self.assertEqual(content['map'], ap[0])
