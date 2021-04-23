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

from common.instance_segmentation_test_case import create_instance_segmentation_export_test_case # pylint: disable=import-error


kwargs = dict(
    problem_name='coco-instance-segmentation',
    ann_file=os.path.dirname(__file__) + '/../../../data/coco/instances_val2017_20imgs.json',
    img_root=os.path.dirname(__file__) + '/../../../data/coco/val2017',
)

class InstanceSegmentation0002TestCase(
        create_instance_segmentation_export_test_case(
            model_name='instance-segmentation-0002',
            **kwargs
        )
):
    """ Test case for instance-segmentation-0002 model export. """


class InstanceSegmentation0091TestCase(
        create_instance_segmentation_export_test_case(
            model_name='instance-segmentation-0091',
            **kwargs
        )
):
    """ Test case for instance-segmentation-0091 model export. """


class InstanceSegmentation0228TestCase(
        create_instance_segmentation_export_test_case(
            model_name='instance-segmentation-0228',
            **kwargs
        )
):
    """ Test case for instance-segmentation-0228 model export. """


class InstanceSegmentation1039TestCase(
        create_instance_segmentation_export_test_case(
            model_name='instance-segmentation-1039',
            **kwargs
        )
):
    """ Test case for instance-segmentation-1039 model export. """


class InstanceSegmentation1040TestCase(
        create_instance_segmentation_export_test_case(
            model_name='instance-segmentation-1040',
            **kwargs
        )
):
    """ Test case for instance-segmentation-1040 model export. """
