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


def download_and_extract_coco_val2017(coco_dir):
    val_dir = os.path.join(coco_dir, 'val2017')
    zip_file = os.path.join(coco_dir, 'val2017.zip')
    link = 'http://images.cocodataset.org/zips/val2017.zip'
    if not os.path.exists(val_dir):
        if not os.path.exists(zip_file):
            run_through_shell(f'wget --no-verbose {link} -P {coco_dir}')
        run_through_shell(f'unzip {zip_file} -d {coco_dir}')


def create_instance_segmentation_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    TestCase = create_test_case('instance_segmentation',
                                **kwargs,
                                metric_keys=['bbox', 'segm'],
                                expected_outputs_dir=expected_outputs_dir)

    class InstanceSegmenationTestCase(TestCase):

        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            coco_dir = os.path.abspath(f'{os.path.dirname(__file__)}/../../../../data/coco')
            download_and_extract_coco_val2017(coco_dir)

    return InstanceSegmenationTestCase


def create_instance_segmentation_export_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    ExportTestCase = create_export_test_case('instance_segmentation',
                                             **kwargs,
                                             metric_keys=['bbox', 'segm'],
                                             expected_outputs_dir=expected_outputs_dir)

    class InstanceSegmenationExportTestCase(ExportTestCase):

        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            coco_dir = os.path.abspath(f'{os.path.dirname(__file__)}/../../../../data/coco')
            download_and_extract_coco_val2017(coco_dir)

    return InstanceSegmenationExportTestCase
