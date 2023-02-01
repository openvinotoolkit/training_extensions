"""
Config Helpers for OTX Training
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

from typing import Tuple

from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.helper import create
from otx.api.entities.model_template import ModelTemplate, parse_model_template


def get_config_and_task_name(template_file_path: str) -> Tuple[ConfigurableParameters, str]:
    """Return configurable parameters and model name given template path

    Args:
        template_file_path (str): template path

    Returns:
        Tuple[ConfigurableParameters, str]: Configurable parameters, model name
    """
    model_template: ModelTemplate = parse_model_template(template_file_path)
    hyper_parameters: dict = model_template.hyper_parameters.data
    config: ConfigurableParameters = create(hyper_parameters)
    return config, model_template.name
