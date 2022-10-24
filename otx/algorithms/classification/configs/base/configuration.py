"""Configuration file of OTX Classification."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=invalid-name

from attr import attrs

from otx.algorithms.common.configs import BaseConfig
from otx.api.configuration.elements import (
    add_parameter_group,
    boolean_attribute,
    string_attribute,
)


@attrs
class ClassificationConfig(BaseConfig):
    """Configurations of classification task."""

    @attrs
    class __LearningParameters(BaseConfig.BaseLearningParameters):
        """Learning parameter configurations."""

        header = string_attribute("Learning Parameters")
        description = header

    @attrs
    class __AlgoBackend(BaseConfig.BaseAlgoBackendParameters):
        """Algorithm backend configurations."""

        header = string_attribute("Parameters for the MPA algo-backend")
        description = header

    @attrs
    class __POTParameter(BaseConfig.BasePOTParameter):
        """POT-related parameter configurations."""

        header = string_attribute("POT Parameters")
        description = header
        visible_in_ui = boolean_attribute(False)

    learning_parameters = add_parameter_group(__LearningParameters)
    algo_backend = add_parameter_group(__AlgoBackend)
    pot_parameters = add_parameter_group(__POTParameter)
