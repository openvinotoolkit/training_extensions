"""Configurable parameters for Padim anomaly Detection task."""

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

from attr import attrs

from otx.algorithms.anomaly.configs.base import BaseAnomalyConfig
from otx.algorithms.anomaly.configs.detection.rkde.configuration_enums import (
    FeatureScalingMethod,
    ROIStage,
)
from otx.api.configuration.elements import (
    add_parameter_group,
    configurable_float,
    configurable_integer,
    selectable,
    string_attribute,
)


@attrs
class RKDEAnomalyDetectionConfig(BaseAnomalyConfig):
    """Configurable parameters for RKDE anomaly Detection task."""

    @attrs
    class LearningParameters(BaseAnomalyConfig.LearningParameters):
        """Parameters that can be tuned using HPO."""

        header = string_attribute("Learning Parameters")
        description = header

        roi_stage = selectable(
            default_value=ROIStage.RCNN,
            header="ROI stage",
            description="Processing stage of the region extractor that yields the box proposals.",
            editable=True,
            visible_in_ui=True,
        )

        roi_score_threshold = configurable_float(
            default_value=0.009,
            header="ROI score threshold",
            min_value=1e-5,
            max_value=1,
            description="Confidence score threshold for the region proposals.",
        )

        iou_threshold = configurable_float(
            default_value=0.3,
            header="IOU threshold",
            min_value=1e-5,
            max_value=1,
            description="Intersection-Over-Union threshold used in class-agnostic NMS when post-processing the raw "
            "region proposals.",
        )

        max_detections_per_image = configurable_integer(
            default_value=100,
            min_value=1,
            max_value=1000,
            header="Max detections per image",
            description="Maximum number of region proposals per image. Region proposals are included/excluded based on "
            " class/objectness score.",
        )

        n_pca_components = configurable_integer(
            default_value=16,
            min_value=2,
            max_value=128,
            header="Number of PCA components.",
            description="Number of principal components used for dimensionality reduction during pre-processing "
            "before training the KDE model.",
        )

        feature_scaling_method = selectable(
            default_value=FeatureScalingMethod.SCALE,
            header="Feature scaling Method",
            description="Feature scaling method applied to the collection of extracted features before applying KDE.",
            editable=True,
            visible_in_ui=True,
        )

    learning_parameters = add_parameter_group(LearningParameters)
