"""Configurable parameters for STFPM anomaly base task."""

# Copyright (C) 2022 Intel Corporation
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

from otx.algorithms.anomaly.configs.base.configuration import BaseAnomalyConfig
from otx.algorithms.anomaly.configs.base.configuration_enums import (
    EarlyStoppingMetrics,
    ModelBackbone,
)
from otx.api.configuration.elements import (
    ParameterGroup,
    add_parameter_group,
    configurable_float,
    configurable_integer,
    selectable,
    string_attribute,
)
from otx.api.configuration.model_lifecycle import ModelLifecycle


@attrs
class STFPMAnomalyBaseConfig(BaseAnomalyConfig):
    """Configurable parameters for STFPM anomaly base task."""

    header = string_attribute("Configuration for STFPM")
    description = header

    @attrs
    class LearningParameters(BaseAnomalyConfig.LearningParameters):
        """Parameters that can be tuned using HPO."""

        lr = configurable_float(
            default_value=0.4,
            header="Learning Rate",
            min_value=1e-3,
            max_value=1,
            description="Learning rate used for optimizing the Student network.",
        )

        momentum = configurable_float(
            default_value=0.9,
            header="Momentum",
            min_value=0.1,
            max_value=1.0,
            description="Momentum used for SGD optimizer",
        )

        weight_decay = configurable_float(
            default_value=0.0001,
            header="Weight Decay",
            min_value=1e-5,
            max_value=1,
            description="Decay for SGD optimizer",
        )

        backbone = selectable(
            default_value=ModelBackbone.RESNET18,
            header="Model Backbone",
            description="Pre-trained backbone used for feature extraction",
        )

        @attrs
        class EarlyStoppingParameters(ParameterGroup):
            """Early stopping parameters."""

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
