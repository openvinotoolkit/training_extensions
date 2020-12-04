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


class InstanceSegmentation0002TestCase(
    create_test_case(
        'coco-instance-segmentation',
        'instance-segmentation-0002',
        os.path.dirname(__file__) + '/../../../data/coco/instances_val2017_20imgs.json',
        os.path.dirname(__file__) + '/../../../data/coco/val2017'
    )
):
    """ Test case for instance-segmentation-0002 model. """


class InstanceSegmentation0091TestCase(
        create_test_case(
            'coco-instance-segmentation',
            'instance-segmentation-0091',
            os.path.dirname(__file__) + '/../../../data/coco/instances_val2017_20imgs.json',
            os.path.dirname(__file__) + '/../../../data/coco/val2017'
        )
):
    """ Test case for instance-segmentation-0091 model. """


class InstanceSegmentation0228TestCase(
        create_test_case(
            'coco-instance-segmentation',
            'instance-segmentation-0228',
            os.path.dirname(__file__) + '/../../../data/coco/instances_val2017_20imgs.json',
            os.path.dirname(__file__) + '/../../../data/coco/val2017'
        )
):
    """ Test case for instance-segmentation-0228 model. """


class InstanceSegmentation0904TestCase(
        create_test_case(
            'coco-instance-segmentation',
            'instance-segmentation-0904',
            os.path.dirname(__file__) + '/../../../data/coco/instances_val2017_20imgs.json',
            os.path.dirname(__file__) + '/../../../data/coco/val2017'
        )
):
    """ Test case for instance-segmentation-0904 model. """


class InstanceSegmentation0912TestCase(
        create_test_case(
            'coco-instance-segmentation',
            'instance-segmentation-0912',
            os.path.dirname(__file__) + '/../../../data/coco/instances_val2017_20imgs.json',
            os.path.dirname(__file__) + '/../../../data/coco/val2017'
        )
):
    """ Test case for instance-segmentation-0912 model. """

