"""Configuration file of OTX Action Tasks."""

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

from otx.algorithms.common.configs import BaseConfig, LearningRateSchedule
from otx.api.configuration.elements import (
    add_parameter_group,
    boolean_attribute,
    selectable,
    string_attribute,
)

# pylint: disable=invalid-name


# TODO Check action detection requires another config
@attrs
class ActionConfig(BaseConfig):
    """Configurations of OTX Action Tasks."""

    header = string_attribute("Configuration for an action classification task of OTX")
    description = header

    @attrs
    class __LearningParameters(BaseConfig.BaseLearningParameters):
        header = string_attribute("Learning Parameters")
        description = header

        learning_rate_schedule = selectable(
            default_value=LearningRateSchedule.COSINE,
            header="Learning rate schedule",
            description="Specify learning rate scheduling for the MMaction task. "
            "When training for a small number of epochs (N < 10), the fixed "
            "schedule is recommended. For training for 10 < N < 25 epochs, "
            "step-wise or exponential annealing might give better results. "
            "Finally, for training on large datasets for at least 20 "
            "epochs, cyclic annealing could result in the best model.",
            editable=True,
            visible_in_ui=True,
        )

    @attrs
    class __NNCFOptimization(BaseConfig.BaseNNCFOptimization):
        header = string_attribute("Optimization by NNCF")
        description = header
        visible_in_ui = boolean_attribute(False)

    @attrs
    class __POTParameter(BaseConfig.BasePOTParameter):
        header = string_attribute("POT Parameters")
        description = header
        visible_in_ui = boolean_attribute(False)

    @attrs
    class __AlgoBackend(BaseConfig.BaseAlgoBackendParameters):
        header = string_attribute("Parameters for the MPA algo-backend")
        description = header

    learning_parameters = add_parameter_group(__LearningParameters)
    nncf_optimization = add_parameter_group(__NNCFOptimization)
    pot_parameters = add_parameter_group(__POTParameter)
    algo_backend = add_parameter_group(__AlgoBackend)
