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

from common.utils import download_if_not_yet, collect_ap


def export_test_case(model_name, snapshot_name=None, alt_ssd_export=False):
    class ExportTestCase(unittest.TestCase):
        def setUp(self):
            self.model_name = model_name
            if snapshot_name is None:
                self.snapshot_name = f'{self.model_name}.pth'
            else:
                self.snapshot_name = snapshot_name

            self.data_folder = '../../data'
            self.work_dir = os.path.join('/tmp/', self.model_name)
            os.makedirs(self.work_dir, exist_ok=True)
            self.configuration_file = f'./configs/{self.model_name}.py'
            os.system(f'cp {self.configuration_file} {self.work_dir}/')
            self.configuration_file = os.path.join(self.work_dir,
                                                   os.path.basename(self.configuration_file))
            self.ote_url = 'https://download.01.org/opencv/openvino_training_extensions'
            self.url = f'{self.ote_url}/models/object_detection/{self.snapshot_name}'
            download_if_not_yet(self.work_dir, self.url)

            self.test_export_thr = 0.01

        def export_test(self, alt_ssd_export, thr):
            if alt_ssd_export:
                export_command_end = '--alt_ssd_export'
                export_dir = os.path.join(self.work_dir, "alt_ssd_export")
                log_file = os.path.join(export_dir, 'test_alt_ssd_export.log')
            else:
                export_dir = os.path.join(self.work_dir, "export")
                log_file = os.path.join(export_dir, 'test_export.log')
                export_command_end = ''

            os.system(
                f'/opt/intel/openvino/bin/setupvars.sh;'
                f'python ../../external/mmdetection/tools/export.py '
                f'{self.configuration_file} '
                f'{os.path.join(self.work_dir, self.snapshot_name)} '
                f'{export_dir} '
                f'openvino {export_command_end};'
                f'python ../../external/mmdetection/tools/test_exported.py '
                f'{self.configuration_file} '
                f'{os.path.join(export_dir, self.model_name + ".xml")} '
                f'--out res.pkl --eval bbox 2>&1 | tee {log_file}')

            ap = collect_ap(log_file)

            with open(f'tests/expected_outputs/{self.model_name}.json') as read_file:
                content = json.load(read_file)

            self.assertGreater(ap[0], content['map'] - thr)

        def test_export(self):
            self.export_test(False, self.test_export_thr)

    class ExportWithAltSsdTestCase(ExportTestCase):

        def setUp(self):
            super().setUp()
            self.test_alt_ssd_export_thr = 0.03

        def test_alt_ssd_export(self):
            self.export_test(True, self.test_alt_ssd_export_thr)

    if alt_ssd_export:
        return ExportWithAltSsdTestCase

    return ExportTestCase
