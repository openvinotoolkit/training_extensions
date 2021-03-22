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

from common.object_detection_test_case import create_object_detection_test_case # pylint: disable=import-error

kwargs = dict(
    problem_name='face-detection',
    ann_file=os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
    img_root=os.path.dirname(__file__) + '/../../../data/airport/'
)

class FaceDetection0200TestCase(
        create_object_detection_test_case(
            model_name='face-detection-0200',
            **kwargs,
        )
):
    """ Test case for face-detection-0200 model. """


class FaceDetection0202TestCase(
        create_object_detection_test_case(
            model_name='face-detection-0202',
            **kwargs,
        )
):
    """ Test case for face-detection-0202 model. """


class FaceDetection0204TestCase(
        create_object_detection_test_case(
            model_name='face-detection-0204',
            **kwargs,
        )
):
    """ Test case for face-detection-0204 model. """


class FaceDetection0205TestCase(
        create_object_detection_test_case(
            model_name='face-detection-0205',
            **kwargs,
        )
):
    """ Test case for face-detection-0205 model. """


class FaceDetection0206TestCase(
        create_object_detection_test_case(
            model_name='face-detection-0206',
            **kwargs,
        )
):
    """ Test case for face-detection-0206 model. """


class FaceDetection0207TestCase(
        create_object_detection_test_case(
            model_name='face-detection-0207',
            **kwargs,
        )
):
    """ Test case for face-detection-0207 model. """
