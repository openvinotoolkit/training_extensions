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

from common.custom_instance_segmentation_test_case import create_custom_instance_segmentation_test_case # pylint: disable=import-error


kwargs = dict(
    problem_name='custom-instance-segmentation'
)


class EfficientnetB2b_MaskRCNN_480x480_TestCase(
        create_custom_instance_segmentation_test_case(
            model_name='efficientnet_b2b-mask_rcnn-480x480',
            **kwargs
        )
):
    """ Test case for efficientnet_b2b-mask_rcnn-480x480 model. """


class EfficientnetB2b_MaskRCNN_576x576_TestCase(
        create_custom_instance_segmentation_test_case(
            model_name='efficientnet_b2b-mask_rcnn-576x576',
            **kwargs
        )
):
    """ Test case for efficientnet_b2b-mask_rcnn-576x576 model. """
