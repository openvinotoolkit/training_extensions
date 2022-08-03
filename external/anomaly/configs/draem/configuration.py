"""
Configurable parameters for Draem anomaly task
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

from attr import attrs
from configs.base import BaseAnomalyConfig
from configs.base.configuration_enums import EarlyStoppingMetrics
from ote_sdk.configuration.elements import (
    ParameterGroup,
    add_parameter_group,
    configurable_float,
    configurable_integer,
    selectable,
    string_attribute,
)
from ote_sdk.configuration.model_lifecycle import ModelLifecycle


@attrs
class DraemAnomalyBaseConfig(BaseAnomalyConfig):
    """
    Configurable parameters for DRAEM anomaly classification task.
    """

    header = string_attribute("Configuration for Draem")
    description = header

    @attrs
    class LearningParameters(BaseAnomalyConfig.LearningParameters):
        """Parameters that can be tuned using HPO."""

        header = string_attribute("Learning Parameters")
        description = header

        train_batch_size = configurable_integer(
            default_value=8,
            min_value=1,
            max_value=512,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        lr = configurable_float(
            default_value=0.0001,
            header="Learning Rate",
            min_value=1e-4,
            max_value=1,
            description="Learning rate used for optimizing the network.",
        )

        @attrs
        class EarlyStoppingParameters(ParameterGroup):
            """
            Early stopping parameters
            """

            header = string_attribute("Early Stopping Parameters")
            description = header

            metric = selectable(
                default_value=EarlyStoppingMetrics.IMAGE_AUROC,
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

        max_epochs = configurable_integer(
            default_value=100,
            header="Max Epochs",
            min_value=1,
            max_value=500,
            description="Maximum number of epochs to train the model for.",
            warning="Training for very few epochs might lead to poor performance. If Early Stopping is enabled then "
            "increasing the value of max epochs might not lead to desired result.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

    learning_parameters = add_parameter_group(LearningParameters)
