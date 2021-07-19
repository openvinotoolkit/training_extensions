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

from common.text_spotting_test_case import create_text_spotting_test_case # pylint: disable=import-error


kwargs = dict(
    problem_name='alphanumeric-text-spotting',
    ann_file=os.path.dirname(__file__) + '/../../../data/horizontal_text_detection/annotation.json',
    img_root=os.path.dirname(__file__) + '/../../../data/horizontal_text_detection/',
)


class TextSpotting0004TestCase(
        create_text_spotting_test_case(
            model_name='text-spotting-0005',
            **kwargs)
):
    """ Test case for text-spotting-0005 model. """
