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
    configurable_boolean,
    configurable_integer,
    string_attribute,
)
from otx.api.configuration.enums import ModelLifecycle


@attrs
class ClassificationConfig(BaseConfig):
    """Configurations of classification task."""

    @attrs
    class __LearningParameters(BaseConfig.BaseLearningParameters):
        """Learning parameter configurations."""

        header = string_attribute("Learning Parameters")
        description = header

        max_num_epochs = configurable_integer(
            default_value=200,
            min_value=1,
            max_value=1000,
            header="Maximum number of training epochs",
            description="Increasing this value causes the results to be more robust but training time "
            "will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        enable_lr_finder = configurable_boolean(
            default_value=False,
            header="Enable automatic learing rate estimation",
            description="Learning rate parameter value will be ignored if enabled.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

    @attrs
    class __AlgoBackend(BaseConfig.BaseAlgoBackendParameters):
        """Algorithm backend configurations."""

        header = string_attribute("Parameters for the OTX algo-backend")
        description = header

        enable_noisy_label_detection = configurable_boolean(
            default_value=False,
            header="Enable loss dynamics tracking for noisy label detection",
            description="Set to True to enable loss dynamics tracking for each sample to detect noisy labeled samples.",
            editable=False,
            visible_in_ui=False,
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

    @attrs
    class __POTParameter(BaseConfig.BasePOTParameter):
        """POT-related parameter configurations."""

        header = string_attribute("POT Parameters")
        description = header
        visible_in_ui = boolean_attribute(False)

    @attrs
    class __NNCFOptimization(BaseConfig.BaseNNCFOptimization):
        header = string_attribute("Optimization by NNCF")
        description = header
        visible_in_ui = boolean_attribute(False)

    learning_parameters = add_parameter_group(__LearningParameters)
    algo_backend = add_parameter_group(__AlgoBackend)
    pot_parameters = add_parameter_group(__POTParameter)
    nncf_optimization = add_parameter_group(__NNCFOptimization)
