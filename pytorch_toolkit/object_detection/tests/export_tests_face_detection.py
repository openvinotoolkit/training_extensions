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

from common.test_case import create_export_test_case


class FaceDetection0200TestCase(
        create_export_test_case(
            'face-detection',
            'face-detection-0200',
            '../../../../../data/airport/annotation_faces_train.json',
            '../../../../../data/airport/',
            True)
):
    """ Test case for face-detection-0200 model export. """


class FaceDetection0202TestCase(
        create_export_test_case(
            'face-detection',
            'face-detection-0202',
            '../../../../../data/airport/annotation_faces_train.json',
            '../../../../../data/airport/',
            True)
):
    """ Test case for face-detection-0202 model export. """


class FaceDetection0204TestCase(
        create_export_test_case(
            'face-detection',
            'face-detection-0204',
            '../../../../../data/airport/annotation_faces_train.json',
            '../../../../../data/airport/',
            True)
):
    """ Test case for face-detection-0204 model export. """


class FaceDetection0205TestCase(
        create_export_test_case(
            'face-detection',
            'face-detection-0205',
            '../../../../../data/airport/annotation_faces_train.json',
            '../../../../../data/airport/',
            False)
):
    """ Test case for face-detection-0205 model export. """


class FaceDetection0206TestCase(
        create_export_test_case(
            'face-detection',
            'face-detection-0206',
            '../../../../../data/airport/annotation_faces_train.json',
            '../../../../../data/airport/',
            False)
):
    """ Test case for face-detection-0206 model export. """


class FaceDetection0207TestCase(
        create_export_test_case(
            'face-detection',
            'face-detection-0207',
            '../../../../../data/airport/annotation_faces_train.json',
            '../../../../../data/airport/',
            True)
):
    """ Test case for face-detection-0207 model export. """
