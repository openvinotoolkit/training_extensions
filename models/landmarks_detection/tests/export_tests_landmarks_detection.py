# Copyright (C) 2021 Intel Corporation
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

from common.landmarks_detection_test_case import create_landmarks_detection_test_case

kwargs = dict(
    problem_name='landmarks-detection',
    ann_file=os.path.dirname(__file__) + '/../../../data/wider_face_lms/wider_lms_test.json',
    img_root=os.path.dirname(__file__) + '/../../../data/wider_face_lms',
)

class LandmarksDetection0001TestCase(
        create_landmarks_detection_test_case(
            model_name='landmarks-detection-0001',
            **kwargs
        )
):
    """ Test case for landmarks-detection-0001 model export. """
