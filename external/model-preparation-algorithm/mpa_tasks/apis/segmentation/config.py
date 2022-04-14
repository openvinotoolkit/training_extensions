# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from attr import attrs

from ote_sdk.configuration.elements import (add_parameter_group,
                                            ParameterGroup,
                                            # configurable_boolean,
                                            configurable_float,
                                            configurable_integer,
                                            selectable,
                                            string_attribute)

from mpa_tasks.apis import BaseConfig, LearningRateSchedule
from ote_sdk.configuration.model_lifecycle import ModelLifecycle
from segmentation_tasks.apis.segmentation.configuration_enums import Models


@attrs
class SegmentationConfig(BaseConfig):
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
    class __Postprocessing(ParameterGroup):
        header = string_attribute("Postprocessing")
        description = header

        class_name = selectable(default_value=Models.BlurSegmentation,
                                header="Model class for inference",
                                description="Model classes with defined pre- and postprocessing",
                                editable=False,
                                visible_in_ui=True)
        blur_strength = configurable_integer(
            header="Blur strength",
            description="With a higher value, the segmentation output will be smoother, but less accurate.",
            default_value=1,
            min_value=1,
            max_value=25,
            affects_outcome_of=ModelLifecycle.INFERENCE
        )
        soft_threshold = configurable_float(
            default_value=0.5,
            header="Soft threshold",
            description="The threshold to apply to the probability output of the model, for each pixel. A higher value "
                        "means a stricter segmentation prediction.",
            min_value=0.0,
            max_value=1.0,
            affects_outcome_of=ModelLifecycle.INFERENCE
        )

    @attrs
    class __AlgoBackend(BaseConfig.BaseAlgoBackendParameters):
        header = string_attribute('Parameters for the MPA algo-backend')
        description = header

    learning_parameters = add_parameter_group(__LearningParameters)
    postprocessing = add_parameter_group(__Postprocessing)
    algo_backend = add_parameter_group(__AlgoBackend)
