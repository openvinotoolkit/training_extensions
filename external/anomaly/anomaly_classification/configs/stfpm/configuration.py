"""
Configurable parameters for STFPM anomaly classification task
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

from anomaly_classification.configs.configuration import \
    BaseAnomalyClassificationConfig
from anomaly_classification.configs.configuration_enums import \
    EarlyStoppingMetrics
from attr import attrs
from ote_sdk.configuration.elements import (ParameterGroup,
                                            add_parameter_group, selectable,
                                            string_attribute)


@attrs
class STFPMConfig(BaseAnomalyClassificationConfig):
    """
    Configurable parameters for STFPM anomaly classification task.
    """

    header = string_attribute("Configuration for STFPM")
    description = header

    @attrs
    class ModelParameters(ParameterGroup):
        header = string_attribute("Model Parameters")
        description = header

        @attrs
        class EarlyStoppingParameters(ParameterGroup):
            header = string_attribute("Early Stopping Parameters")
            description = header

            metric = selectable(
                default_value=EarlyStoppingMetrics.IMAGE_ROC_AUC,
                header="Early Stopping Metric",
                description="The metric used to determine if the model should stop training",
            )

        early_stopping = add_parameter_group(EarlyStoppingParameters)

    model = add_parameter_group(ModelParameters)
