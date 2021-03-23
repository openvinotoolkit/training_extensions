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

from common.image_classification_test_case import create_image_classification_export_test_case # pylint: disable=import-error


kwargs = dict(
    problem_name='imagenet-classification',
    ann_file='',
    img_root=''
)


class CustomClassificationExportTestCaseLarge1(
        create_image_classification_export_test_case(
            model_name='mobilenet-v3-large-1',
            **kwargs,
        )
):
    """ Test case for mobilenet v3 large x1.0 model export. """


class CustomClassificationExportTestCaseLarge75(
        create_image_classification_export_test_case(
            model_name='mobilenet-v3-large-075',
            **kwargs,
        )
):
    """ Test case for mobilenet v3 large x0.75 model export. """


class CustomClassificationExportTestCaseSmall(
        create_image_classification_export_test_case(
            model_name='mobilenet-v3-small',
            **kwargs,
        )
):
    """ Test case for mobilenet v3 small model export. """


class CustomClassificationExportTestCaseEfficientnet(
        create_image_classification_export_test_case(
            model_name='efficientnet-b0',
            **kwargs,
        )
):
    """ Test case for efficientnet model export. """
