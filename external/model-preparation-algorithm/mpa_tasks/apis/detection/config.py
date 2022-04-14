# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from attr import attrs

from ote_sdk.configuration.elements import (add_parameter_group,
                                            # ParameterGroup,
                                            # configurable_boolean,
                                            # configurable_float,
                                            # configurable_integer,
                                            selectable,
                                            string_attribute)

from mpa_tasks.apis import BaseConfig, LearningRateSchedule


@attrs
class DetectionConfig(BaseConfig):
    header = string_attribute("Configuration for an object detection task of MPA")
    description = header

    @attrs
    class __LearningParameters(BaseConfig.BaseLearningParameters):
        header = string_attribute('Learning Parameters')
        description = header

        learning_rate_schedule = selectable(
            default_value=LearningRateSchedule.COSINE,
            header='Learning rate schedule',
            description='Specify learning rate scheduling for the MMDetection task. '
                        'When training for a small number of epochs (N < 10), the fixed '
                        'schedule is recommended. For training for 10 < N < 25 epochs, '
                        'step-wise or exponential annealing might give better results. '
                        'Finally, for training on large datasets for at least 20 '
                        'epochs, cyclic annealing could result in the best model.',
            editable=True, visible_in_ui=True)

    @attrs
    class __Postprocessing(BaseConfig.BasePostprocessing):
        header = string_attribute("Postprocessing")
        description = header

    @attrs
    class __NNCFOptimization(BaseConfig.BaseNNCFOptimization):
        header = string_attribute("Optimization by NNCF")
        description = header

    @attrs
    class __POTParameter(BaseConfig.BasePOTParameter):
        header = string_attribute("POT Parameters")
        description = header

    @attrs
    class __AlgoBackend(BaseConfig.BaseAlgoBackendParameters):
        header = string_attribute('Parameters for the MPA algo-backend')
        description = header

    learning_parameters = add_parameter_group(__LearningParameters)
    postprocessing = add_parameter_group(__Postprocessing)
    nncf_optimization = add_parameter_group(__NNCFOptimization)
    pot_parameters = add_parameter_group(__POTParameter)
    algo_backend = add_parameter_group(__AlgoBackend)
