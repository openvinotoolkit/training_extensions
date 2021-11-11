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

from sys import maxsize

from attr import attrs
from ote_sdk.configuration import ConfigurableParameters
from ote_sdk.configuration.elements import (
    ParameterGroup,
    add_parameter_group,
    configurable_integer,
    selectable,
    string_attribute,
)
from ote_sdk.configuration.model_lifecycle import ModelLifecycle

from anomaly_classification.configs.configuration_enums import POTQuantizationPreset


@attrs
class AnomalyClassificationConfig(ConfigurableParameters):
    """
    Base OTE configurable parameters for anomaly classification task.
    """

    header = string_attribute("Configuration for an anomaly classification task")
    description = header

    @attrs
    class DatasetParameters(ParameterGroup):
        header = string_attribute("Dataset Parameters")
        description = header

        train_batch_size = configurable_integer(
            default_value=5,
            min_value=1,
            max_value=512,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        num_workers = configurable_integer(
            default_value=8,
            min_value=0,
            max_value=8,
            header="Number of workers",
            description="Increasing this value might improve training speed however it might cause out of memory "
            "errors. If the number of workers is set to zero, data loading will happen in the main "
            "training thread.",
        )

    @attrs
    class POTParameters(ParameterGroup):
        header = string_attribute("POT Parameters")
        description = header

        preset = selectable(
            default_value=POTQuantizationPreset.PERFORMANCE,
            header="Preset",
            description="Quantization preset that defines quantization scheme",
        )

        stat_subset_size = configurable_integer(
            header="Number of data samples",
            description="Number of data samples used for post-training optimization",
            default_value=300,
            min_value=1,
            max_value=maxsize,
        )

    dataset = add_parameter_group(DatasetParameters)
    pot_parameters = add_parameter_group(POTParameters)
