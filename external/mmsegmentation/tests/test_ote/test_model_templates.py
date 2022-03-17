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

import glob
import os
import unittest

from e2e_test_system import e2e_pytest_api
from ote_sdk.entities.model_template import parse_model_template


def gen_parse_model_template_tests():
    class ModelTemplateTests(unittest.TestCase):
        pass

    base_dir = os.path.join('configs', 'ote')
    glob_path = os.path.join(base_dir, "**", "template.yaml")
    templates = glob.glob(glob_path, recursive=True)
    for template in templates:
        path = os.path.relpath(template)

        @e2e_pytest_api
        def test_template(self, path=path):
            template = parse_model_template(path)
            assert template.hyper_parameters.data

        setattr(ModelTemplateTests, 'test_' + path.replace(' ', '_').replace('/', '_'), test_template)

    return ModelTemplateTests


class TestParseModelTemplates(gen_parse_model_template_tests()):
    pass
