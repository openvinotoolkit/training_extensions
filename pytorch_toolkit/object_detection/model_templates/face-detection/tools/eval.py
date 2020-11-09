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

import argparse
import os

from ote import MODEL_TEMPLATE_FILENAME
from ote.api import test_args_parser
from ote.args_conversion import convert_ote_to_oteod_test_args
from ote.evaluation.face_detection import evaluate


def parse_args(template_filename):
    """ Parses input args. """

    parser = argparse.ArgumentParser(parents=[test_args_parser(template_filename)], add_help=False)
    parser.add_argument('--wider-dir',
                        help='Location of WiderFace dataset.',
                        default='data/wider_face')
    return parser.parse_args()

ote_args = vars(parse_args(MODEL_TEMPLATE_FILENAME))
oteod_args = convert_ote_to_oteod_test_args(os.path.dirname(MODEL_TEMPLATE_FILENAME), ote_args)
evaluate(**oteod_args, wider_dir=ote_args['wider_dir'])
