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
from ote_sdk.configuration.elements import (ParameterGroup,
                                            add_parameter_group,
                                            string_attribute,
                                            selectable)

from anomaly_classification.configs.configuration import AnomalyClassificationConfig
from anomaly_classification.configs.configuration_enums import ModelName


@attrs
class PadimConfig(AnomalyClassificationConfig):
    """
    Configurable parameters for PADIM anomaly classification task.
    """
    header = string_attribute("Configuration for padim")
    description = header

    @attrs
    class ModelParameters(ParameterGroup):
        header = string_attribute("Model Parameters")
        description = header

        name = selectable(
            default_value=ModelName.PADIM,
            header="Model Name",
            description="Name of the model that should be used for training.",
            editable=False,
            visible_in_ui=False
        )

    model = add_parameter_group(ModelParameters)
