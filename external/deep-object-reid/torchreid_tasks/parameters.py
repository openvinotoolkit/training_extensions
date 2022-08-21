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
from sys import maxsize

from ote_sdk.configuration.elements import (ParameterGroup,
                                            add_parameter_group,
                                            boolean_attribute,
                                            configurable_boolean,
                                            configurable_float,
                                            configurable_integer,
                                            selectable,
                                            string_attribute,
                                            )
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.configuration.enums import ModelLifecycle, AutoHPOState

from .parameters_enums import POTQuantizationPreset

@attrs
class OTEClassificationParameters(ConfigurableParameters):
    header = string_attribute("Configuration for an image classification task")
    description = header

    @attrs
    class __LearningParameters(ParameterGroup):
        header = string_attribute("Learning Parameters")
        description = header

        batch_size = configurable_integer(
            default_value=32,
            min_value=1,
            max_value=512,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING,
            auto_hpo_state=AutoHPOState.NOT_POSSIBLE
        )

        max_num_epochs = configurable_integer(
            default_value=200,
            min_value=1,
            max_value=1000,
            header="Maximum number of training epochs",
            description="Increasing this value causes the results to be more robust but training time "
            "will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        learning_rate = configurable_float(
            default_value=0.01,
            min_value=1e-07,
            max_value=1e-01,
            header="Learning rate",
            description="Increasing this value will speed up training \
                         convergence but might make it unstable.",
            affects_outcome_of=ModelLifecycle.TRAINING,
            auto_hpo_state=AutoHPOState.NOT_POSSIBLE
        )

        enable_lr_finder = configurable_boolean(
            default_value=False,
            header="Enable automatic learing rate estimation",
            description="Learning rate parameter value will be ignored if enabled.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        enable_early_stopping = configurable_boolean(
            default_value=True,
            header="Enable adaptive early stopping of the training",
            description="Adaptive early exit from training when accuracy isn't \
                         changed or decreased for several epochs.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

    @attrs
    class __NNCFOptimization(ParameterGroup):
        header = string_attribute("Optimization by NNCF")
        description = header
        visible_in_ui = boolean_attribute(False)

        enable_quantization = configurable_boolean(
            default_value=True,
            header="Enable quantization algorithm",
            description="Enable quantization algorithm",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        enable_pruning = configurable_boolean(
            default_value=False,
            header="Enable filter pruning algorithm",
            description="Enable filter pruning algorithm",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        pruning_supported = configurable_boolean(
            default_value=False,
            header="Whether filter pruning is supported",
            description="Whether filter pruning is supported",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        maximal_accuracy_degradation = configurable_float(
            default_value=1.0,
            min_value=0.0,
            max_value=100.0,
            header="Maximum accuracy degradation",
            description="The maximal allowed accuracy metric drop",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

    @attrs
    class __POTParameter(ParameterGroup):
        header = string_attribute("POT Parameters")
        description = header
        visible_in_ui = boolean_attribute(False)

        stat_subset_size = configurable_integer(
            header="Number of data samples",
            description="Number of data samples used for post-training optimization",
            default_value=300,
            min_value=1,
            max_value=maxsize
        )

        preset = selectable(default_value=POTQuantizationPreset.PERFORMANCE, header="Preset",
                            description="Quantization preset that defines quantization scheme",
                            editable=False, visible_in_ui=False)

    learning_parameters = add_parameter_group(__LearningParameters)
    nncf_optimization = add_parameter_group(__NNCFOptimization)
    pot_parameters = add_parameter_group(__POTParameter)
