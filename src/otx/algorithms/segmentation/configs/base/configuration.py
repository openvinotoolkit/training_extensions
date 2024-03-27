"""Configuration file of OTX Segmentation."""

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

from otx.algorithms.common.configs import (
    BaseConfig,
    LearningRateSchedule,
    POTQuantizationPreset,
)
from otx.api.configuration.elements import (
    add_parameter_group,
    boolean_attribute,
    configurable_boolean,
    configurable_float,
    configurable_integer,
    selectable,
    string_attribute,
)
from otx.api.configuration.model_lifecycle import ModelLifecycle

# pylint: disable=invalid-name


@attrs
class SegmentationConfig(BaseConfig):
    """Configurations of OTX Segmentation."""

    header = string_attribute("Configuration for an object semantic segmentation task of OTX")
    description = header

    @attrs
    class __LearningParameters(BaseConfig.BaseLearningParameters):
        header = string_attribute("Learning Parameters")
        description = header

        learning_rate_schedule = selectable(
            default_value=LearningRateSchedule.COSINE,
            header="Learning rate schedule",
            description="Specify learning rate scheduling for the MMDetection task. "
            "When training for a small number of epochs (N < 10), the fixed "
            "schedule is recommended. For training for 10 < N < 25 epochs, "
            "step-wise or exponential annealing might give better results. "
            "Finally, for training on large datasets for at least 20 "
            "epochs, cyclic annealing could result in the best model.",
            editable=True,
            visible_in_ui=True,
        )

    @attrs
    class __AlgoBackend(BaseConfig.BaseAlgoBackendParameters):
        header = string_attribute("Parameters for the OTX algo-backend")
        description = header

    @attrs
    class __Postprocessing(BaseConfig.BasePostprocessing):
        header = string_attribute("Postprocessing")
        description = header

        blur_strength = configurable_integer(
            header="Blur strength",
            description="With a higher value, the segmentation output will be smoother, but less accurate.",
            default_value=1,
            min_value=1,
            max_value=25,
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )
        soft_threshold = configurable_float(
            default_value=0.5,
            header="Soft threshold",
            description="The threshold to apply to the probability output of the model, for each pixel. A higher value "
            "means a stricter segmentation prediction.",
            min_value=0.0,
            max_value=1.0,
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )

    @attrs
    class __POTParameter(BaseConfig.BasePOTParameter):
        header = string_attribute("POT Parameters")
        description = header
        visible_in_ui = boolean_attribute(False)

        stat_subset_size = configurable_integer(
            header="Number of data samples",
            description="Number of data samples used for post-training optimization",
            default_value=300,
            min_value=1,
            max_value=1000,
        )

        preset = selectable(
            default_value=POTQuantizationPreset.PERFORMANCE,
            header="Preset",
            description="Quantization preset that defines quantization scheme",
            editable=False,
            visible_in_ui=False,
        )

    @attrs
    class __NNCFOptimization(BaseConfig.BaseNNCFOptimization):
        header = string_attribute("Optimization by NNCF")
        description = header
        visible_in_ui = boolean_attribute(False)

        enable_quantization = configurable_boolean(
            default_value=True,
            header="Enable quantization algorithm",
            description="Enable quantization algorithm",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        enable_pruning = configurable_boolean(
            default_value=False,
            header="Enable filter pruning algorithm",
            description="Enable filter pruning algorithm",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        pruning_supported = configurable_boolean(
            default_value=False,
            header="Whether filter pruning is supported",
            description="Whether filter pruning is supported",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        maximal_accuracy_degradation = configurable_float(
            default_value=1.0,
            min_value=0.0,
            max_value=100.0,
            header="Maximum accuracy degradation",
            description="The maximal allowed accuracy metric drop",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

    learning_parameters = add_parameter_group(__LearningParameters)
    postprocessing = add_parameter_group(__Postprocessing)
    algo_backend = add_parameter_group(__AlgoBackend)
    nncf_optimization = add_parameter_group(__NNCFOptimization)
    pot_parameters = add_parameter_group(__POTParameter)
