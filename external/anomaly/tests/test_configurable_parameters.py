"""
Test configurable parameters for the anomaly classification task
"""

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

import logging
import os

import pytest
from ote_sdk.configuration.helper import convert, create
from ote_sdk.entities.model_template import parse_model_template

from tests.helpers.config import get_config_and_task_name
from anomaly_classification.configs.padim import PadimConfig
from anomaly_classification.configs.stfpm import STFPMConfig
from ote_anomalib.config import get_anomalib_config

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(["model_name", "configurable_parameters"], [("padim", PadimConfig), ("stfpm", STFPMConfig)])
def test_configuration_yaml(configurable_parameters, model_name):
    # assert that we can parse the template.yaml
    template_file_path = os.path.join("anomaly_classification", "configs", model_name, "template.yaml")
    configuration_yaml_loaded, task_name = get_config_and_task_name(template_file_path)

    configuration = configurable_parameters()
    # assert that we can convert our config object to yaml format
    configuration_yaml_str = convert(configuration, str)
    # assert that we can create configurable parameters from the yaml string
    configuration_yaml_converted = create(configuration_yaml_str)
    # assert that we generate an anomalib config from the
    get_anomalib_config(task_name, configuration_yaml_converted)
    # assert that the python class and the yaml file result in the same configurable parameters object
    assert configuration_yaml_converted == configuration_yaml_loaded
