"""Configuration file of OTX Visual Prompting."""

# Copyright (C) 2023 Intel Corporation
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


from attr import attrs

from otx.algorithms.common.configs import BaseConfig, LearningRateSchedule
from otx.api.configuration.elements import (
    add_parameter_group,
    boolean_attribute,
    configurable_integer,
    selectable,
    string_attribute,
)
from otx.api.configuration.model_lifecycle import ModelLifecycle


@attrs
class VisualPromptingConfig(BaseConfig):
    """Configurations of OTX Visual Prompting."""

    header = string_attribute("Configuration for a visual prompting task of OTX")
    description = header

    learning_parameters = add_parameter_group(BaseConfig.BaseLearningParameters)
