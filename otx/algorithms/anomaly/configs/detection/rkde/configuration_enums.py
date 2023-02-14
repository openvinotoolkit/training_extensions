"""Configuration Enums for RKDE Model.

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


class ROIStage(ConfigurableEnum):
    """Region-of-Interest extraction stage.

    This Enum represents the ROI stage used by the region extraction submodule of the RKDE model.
    """

    RCNN = "rcnn"
    RPN = "rpn"


class FeatureScalingMethod(ConfigurableEnum):
    """Feature scaling method.

    This Enum represents the scaling method applied to the extracted features before KDE.
    Scale: Scale to the highest feature vector lenght observed in the training data.
    Norm: Normalize each feature vector to unit length.
    """

    SCALE = "scale"
    NORM = "norm"
