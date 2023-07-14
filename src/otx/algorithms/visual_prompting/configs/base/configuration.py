"""Configuration file of OTX Visual Prompting."""

# Copyright (C) 2023 Intel Corporation
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

from otx.algorithms.common.configs import BaseConfig, POTQuantizationPreset
from otx.api.configuration.elements import (
    ParameterGroup,
    add_parameter_group,
    boolean_attribute,
    configurable_boolean,
    configurable_float,
    configurable_integer,
    selectable,
    string_attribute,
)
from otx.api.configuration.model_lifecycle import ModelLifecycle


@attrs
class VisualPromptingBaseConfig(BaseConfig):
    """Configurations of OTX Visual Prompting."""

    header = string_attribute("Configuration for a visual prompting task of OTX")
    description = header

    @attrs
    class __LearningParameters(BaseConfig.BaseLearningParameters):
        header = string_attribute("Learning Parameters")
        description = header

    @attrs
    class __Postprocessing(ParameterGroup):
        header = string_attribute("Postprocessing")
        description = header

        image_size = configurable_integer(
            header="Image size",
            description="The size of the input image to the model.",
            default_value=1024,
            min_value=0,
            max_value=2048,
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )

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

        embedded_processing = configurable_boolean(
            default_value=True,
            header="Embedded processing",
            description="Flag that pre/postprocessing embedded.",
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )

        orig_width = configurable_float(
            header="Original width",
            description="Model input width before embedding processing.",
            default_value=64.0,
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )

        orig_height = configurable_float(
            header="Original height",
            description="Model input height before embedding processing.",
            default_value=64.0,
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )

    @attrs
    class __POTParameter(BaseConfig.BasePOTParameter):
        header = string_attribute("POT Parameters")
        description = header
        visible_in_ui = boolean_attribute(False)

        preset = selectable(
            default_value=POTQuantizationPreset.MIXED,
            header="Preset",
            description="Quantization preset that defines quantization scheme",
            editable=True,
            visible_in_ui=True,
        )

    learning_parameters = add_parameter_group(__LearningParameters)
    postprocessing = add_parameter_group(__Postprocessing)
    pot_parameters = add_parameter_group(__POTParameter)
