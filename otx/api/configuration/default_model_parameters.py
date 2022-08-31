"""This module contains a default set of configurable parameters for a model."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from attr import attrib, attrs

from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.elements import (
    ParameterGroup,
    add_parameter_group,
    boolean_attribute,
    configurable_float,
    configurable_integer,
    string_attribute,
)
from otx.api.configuration.enums.model_lifecycle import ModelLifecycle


@attrs
class DefaultModelParameters(ConfigurableParameters):
    """Configuration element representing a the default set of hyper parameters for a model.

    Attributes:
        header (str): Name of parameter group
        description (str): User friendly string describing what the ModelConfig represents, that will be displayed in
            the UI.
    """

    header: str = attrib(default="Default model hyper parameters")
    description: str = attrib(default="Default model hyper parameter section description", kw_only=True)

    @attrs
    class _LearningParameters(ParameterGroup):
        # Set defaults for the learning parameters. Learning parameters consist of at
        # least batch_size, epochs and learning_rate. These correspond to the 'basic'
        # hyper parameters in model template
        header = string_attribute("Learning Parameters")
        description = string_attribute("Parameters to control basic training behavior.")
        visible_in_ui = boolean_attribute(True)

        batch_size = configurable_integer(
            header="Batch size",
            description="The number of training samples seen in each "
            "iteration of training. Setting this higher will "
            "make the training more stable, but will require "
            "more memory. Setting this lower will make the "
            "training less stable, but will require less "
            "memory.",
            warning="Increasing this value may cause the system to use "
            "more memory than available, potentially causing out "
            "of memory errors, please update with caution.",
            min_value=1,
            max_value=1000,
            default_value=4,
            affects_outcome_of=ModelLifecycle.TRAINING,
        )
        epochs = configurable_integer(
            header="Number of epochs",
            default_value=10,
            min_value=1,
            max_value=10000,
            description="Increasing this value causes the results to be more "
            "robust but training time will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )
        learning_rate = configurable_float(
            header="Learning rate",
            default_value=1e-3,
            min_value=1e-30,
            max_value=1e10,
            description="Increasing this value will speed up training " "convergence but might make it unstable.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

    learning_parameters = add_parameter_group(_LearningParameters)
