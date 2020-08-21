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

from ote import MODEL_TEMPLATE_FILENAME
from ote.api import test_args_parser
from oteod.args_conversion import convert_ote_to_oteod_test_args
from oteod.evaluation.horizontal_text_detection import evaluate

ote_args = vars(test_args_parser(MODEL_TEMPLATE_FILENAME).parse_args())
oteod_args = convert_ote_to_oteod_test_args(os.path.dirname(MODEL_TEMPLATE_FILENAME), ote_args)
evaluate(**oteod_args)

