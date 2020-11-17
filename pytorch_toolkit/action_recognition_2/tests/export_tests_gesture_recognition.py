"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os

from common.test_case import create_export_test_case


class GestureRecognitionJesterTestCase(
        create_export_test_case(
            'gesture-recognition',
            's3d-rgb-mobilenet-v3-stream-jester',
            'train.txt',
            os.path.dirname(__file__) + '/../../../data/gesture_recognition'
        )
):
    """ Test case for s3d-rgb-mobilenet-v3-stream-jester model export. """


class GestureRecognitionASLTestCase(
        create_export_test_case(
            'gesture-recognition',
            's3d-rgb-mobilenet-v3-stream-msasl',
            'train.txt',
            os.path.dirname(__file__) + '/../../../data/gesture_recognition'
        )
):
    """ Test case for s3d-rgb-mobilenet-v3-stream-msasl model export. """
