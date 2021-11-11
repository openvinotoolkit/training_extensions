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
from anomaly_classification.configs.padim import PadimConfig
from anomaly_classification.configs.stfpm import STFPMConfig
from ote_anomalib.config import get_anomalib_config
from ote_sdk.configuration.helper import convert, create

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(["configurable_parameters"], [(PadimConfig,), (STFPMConfig,)])
def test_configuration_yaml(configurable_parameters):
    configuration = configurable_parameters()
    # assert that we can convert our config object to yaml format
    configuration_yaml_str = convert(configuration, str)
    # assert that we can create a SC config object from the yaml string
    configuration_yaml_converted = create(configuration_yaml_str)
    # assert that we generate an anomalib config from the
    get_anomalib_config(configuration_yaml_converted)
    # assert that the python class and the yaml file result in the same configurable parameters object
    model_name = configuration_yaml_converted.model.name.value
    configuration_yaml_loaded = create(
        os.path.join("anomaly_classification", "configs", model_name, "configuration.yaml")
    )
    assert configuration_yaml_converted == configuration_yaml_loaded
