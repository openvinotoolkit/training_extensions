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

from common.test_case import create_export_test_case


class VehicleDetection0200TestCase(
        create_export_test_case(
            'vehicle-detection',
            'vehicle-detection-0200',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/annotation_val.json',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/val',
            True)
):
    """ Test case for vehicle-detection-0200 model export. """


class VehicleDetection0201TestCase(
        create_export_test_case(
            'vehicle-detection',
            'vehicle-detection-0201',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/annotation_val.json',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/val',
            True)
):
    """ Test case for vehicle-detection-0201 model export. """


class VehicleDetection0202TestCase(
        create_export_test_case(
            'vehicle-detection',
            'vehicle-detection-0202',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/annotation_val.json',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/val',
            True)
):
    """ Test case for vehicle-detection-0202 model export. """


class VehicleDetection0203TestCase(
        create_export_test_case(
            'vehicle-detection',
            'vehicle-detection-0203',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/vehicle_annotation_val.json',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/val',
            False)
):
    """ Test case for vehicle-detection-0230 model export. """


class VehicleDetection0204TestCase(
        create_export_test_case(
            'vehicle-detection',
            'vehicle-detection-0204',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/annotation_val.json',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/val',
            False)
):
    """ Test case for vehicle-detection-0204 model export. """
