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

from common.test_case import create_test_case


class VehicleDetection0200TestCase(
        create_test_case(
            'vehicle-detection',
            'vehicle-detection-0200',
            os.path.dirname(__file__) + '../../../data/crossroad_vehicle/annotation_train.json',
            os.path.dirname(__file__) + '../../../data/crossroad_vehicle/train'
        )
):
    """ Test case for vehicle-detection-0200 model. """


class VehicleDetection0201TestCase(
        create_test_case(
            'vehicle-detection',
            'vehicle-detection-0201',
            os.path.dirname(__file__) + '../../../data/crossroad_vehicle/annotation_train.json',
            os.path.dirname(__file__) + '../../../data/crossroad_vehicle/train'
        )
):
    """ Test case for vehicle-detection-0201 model. """


class VehicleDetection0202TestCase(
        create_test_case(
            'vehicle-detection',
            'vehicle-detection-0202',
            os.path.dirname(__file__) + '../../../data/crossroad_vehicle/annotation_train.json',
            os.path.dirname(__file__) + '../../../data/crossroad_vehicle/train'
        )
):
    """ Test case for vehicle-detection-0202 model. """

class VehicleDetection0203TestCase(
        create_test_case(
            'vehicle-detection',
            'vehicle-detection-0203',
            os.path.dirname(__file__) + '../../../data/crossroad_vehicle/annotation_train.json',
            os.path.dirname(__file__) + '../../../data/crossroad_vehicle/train'
        )
):
    """ Test case for vehicle-detection-0203 model. """


class VehicleDetection0204TestCase(
        create_test_case(
            'vehicle-detection',
            'vehicle-detection-0204',
            os.path.dirname(__file__) + '../../../data/crossroad_vehicle/annotation_train.json',
            os.path.dirname(__file__) + '../../../data/crossroad_vehicle/train'
        )
):
    """ Test case for vehicle-detection-0204 model. """
