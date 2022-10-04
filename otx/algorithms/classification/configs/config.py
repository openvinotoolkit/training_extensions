# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from attr import attrs
from otx.algorithms.common.configs import BaseConfig
from otx.api.configuration.elements import add_parameter_group, string_attribute


@attrs
class ClassificationConfig(BaseConfig):
    @attrs
    class __LearningParameters(BaseConfig.BaseLearningParameters):
        header = string_attribute("Learning Parameters")
        description = header

    @attrs
    class __AlgoBackend(BaseConfig.BaseAlgoBackendParameters):
        header = string_attribute("Parameters for the MPA algo-backend")
        description = header

    learning_parameters = add_parameter_group(__LearningParameters)
    algo_backend = add_parameter_group(__AlgoBackend)
