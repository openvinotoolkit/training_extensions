"""Configurable parameters for Padim anomaly task."""

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

from attr import attrs
from configs.base.configuration import BaseAnomalyConfig
from configs.base.configuration_enums import ModelBackbone
from ote_sdk.configuration.elements import (
    add_parameter_group,
    selectable,
    string_attribute,
)


@attrs
class PadimAnomalyBaseConfig(BaseAnomalyConfig):
    """Configurable parameters for PADIM anomaly classification task."""

    header = string_attribute("Configuration for Padim")
    description = header

    @attrs
    class LearningParameters(BaseAnomalyConfig.LearningParameters):
        """Parameters that can be tuned using HPO."""

        header = string_attribute("Learning Parameters")
        description = header

        backbone = selectable(
            default_value=ModelBackbone.RESNET18,
            header="Model Backbone",
            description="Pre-trained backbone used for feature extraction",
        )

    learning_parameters = add_parameter_group(LearningParameters)
