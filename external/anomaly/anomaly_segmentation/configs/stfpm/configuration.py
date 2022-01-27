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

from anomaly_classification.configs.configuration import BaseAnomalyClassificationConfig
from anomaly_classification.configs.configuration_enums import EarlyStoppingMetrics
from attr import attrs
from ote_sdk.configuration.elements import (
    ParameterGroup,
    add_parameter_group,
    configurable_integer,
    selectable,
    string_attribute,
)
from ote_sdk.configuration.model_lifecycle import ModelLifecycle


@attrs
class STFPMConfig(BaseAnomalyClassificationConfig):
    """
    Configurable parameters for STFPM anomaly classification task.
    """

    header = string_attribute("Configuration for STFPM")
    description = header

    @attrs
    class ModelParameters(ParameterGroup):
        """
        Parameter Group for training model
        """

        header = string_attribute("Model Parameters")
        description = header

        @attrs
        class EarlyStoppingParameters(ParameterGroup):
            """
            Early stopping parameters
            """

            header = string_attribute("Early Stopping Parameters")
            description = header

            metric = selectable(
                default_value=EarlyStoppingMetrics.IMAGE_F1,
                header="Early Stopping Metric",
                description="The metric used to determine if the model should stop training",
            )

            patience = configurable_integer(
                default_value=10,
                min_value=1,
                max_value=100,
                header="Early Stopping Patience",
                description="Number of epochs to wait for an improvement in the monitored metric. If the metric has "
                "not improved for this many epochs, the training will stop and the best model will be "
                "returned.",
                warning="Setting this value too low might lead to underfitting. Setting the value too high will "
                "increase the training time and might lead to overfitting.",
                affects_outcome_of=ModelLifecycle.TRAINING,
            )

        early_stopping = add_parameter_group(EarlyStoppingParameters)

    model = add_parameter_group(ModelParameters)
