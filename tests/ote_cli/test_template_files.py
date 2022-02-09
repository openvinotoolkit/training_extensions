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

import os
import pytest

from ote_sdk.entities.model_template import parse_model_template

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component

from ote_cli.registry import Registry

templates = Registry('external').templates
paths = [os.path.relpath(template.model_template_path) for template in templates]
ids = [os.path.relpath(template.model_template_id) for template in templates]

class TestTemplates:
    @e2e_pytest_component
    @pytest.mark.parametrize("path", paths, ids=ids)
    def test_template(self, path):
        template = parse_model_template(path)
        assert template.hyper_parameters.data
