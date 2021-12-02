"""
Enums needed to define the options of selectable parameters in the configurable parameter classes
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

from ote_sdk.configuration import ConfigurableEnum


class POTQuantizationPreset(ConfigurableEnum):
    """
    This Enum represents the quantization preset for post training optimization
    """

    PERFORMANCE = "Performance"
    MIXED = "Mixed"


class EarlyStoppingMetrics(ConfigurableEnum):
    """
    This enum represents the different metrics that can be used for early stopping
    """

    IMAGE_ROC_AUC = "image_roc_auc"
    IMAGE_F1 = "image-f1-score"


class ModelName(ConfigurableEnum):
    """
    This enum represents the different model architectures for anomaly classification
    """

    STFPM = "stfpm"
    PADIM = "padim"


class Inference(ConfigurableEnum):
    """
    This Enum represents the types of pre- and postprocessing for models
    """

    ANOMALY_CLASSIFICATION = "anomaly_classification"
