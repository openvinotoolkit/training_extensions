"""Configuration Enums.

Enums needed to define the options of selectable parameters in the configurable
parameter classes.
"""

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

from otx.api.configuration import ConfigurableEnum


class POTQuantizationPreset(ConfigurableEnum):
    """POT Quantization Preset Enum.

    This Enum represents the quantization preset for post training optimization.
    """

    PERFORMANCE = "Performance"
    MIXED = "Mixed"


class EarlyStoppingMetrics(ConfigurableEnum):
    """Early Stopping Metric Enum.

    This enum represents the different metrics that can be used for early
    stopping.
    """

    IMAGE_ROC_AUC = "image_AUROC"
    IMAGE_F1 = "image_F1Score"


class ModelName(ConfigurableEnum):
    """Model Name Enum.

    This enum represents the different model architectures for anomaly
    classification.
    """

    STFPM = "stfpm"
    PADIM = "padim"


class ModelBackbone(ConfigurableEnum):
    """Model Backbone Enum.

    This enum represents the common backbones that can be used with Padim and
    STFPM.
    """

    RESNET18 = "resnet18"
    WIDE_RESNET_50 = "wide_resnet50_2"
