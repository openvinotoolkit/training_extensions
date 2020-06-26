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

from common.test_case import export_test_case
from common.utils import replace_text_in_file


def text_detection_test_case(model_name):
    class ExportTestCase(export_test_case('horizontal-text-detection', model_name)):
        def setUp(self):
            super().setUp()

            assert replace_text_in_file(self.configuration_file, 'data/text-dataset/',
                                        '../../data/horizontal_text_detection/')
            assert replace_text_in_file(self.configuration_file, 'IC13TEST.json',
                                        'annotation.json')

    return ExportTestCase


class HorizontalTextDetection0001TestCase(text_detection_test_case('horizontal-text-detection-0001')):
    """ Test case for horizontal-text-detection-0001 model export. """
