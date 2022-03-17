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

import os.path as osp

from e2e_test_system import e2e_pytest_api
from ote_sdk.configuration.helper import convert, create

from segmentation_tasks.apis.segmentation.configuration import OTESegmentationConfig


@e2e_pytest_api
def test_configuration_yaml():
    configuration = OTESegmentationConfig()
    configuration_yaml_str = convert(configuration, str)
    configuration_yaml_converted = create(configuration_yaml_str)
    configuration_yaml_loaded = create(osp.join('segmentation_tasks', 'apis', 'segmentation', 'configuration.yaml'))
    assert configuration_yaml_converted == configuration_yaml_loaded
