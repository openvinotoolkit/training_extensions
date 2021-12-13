"""
AnomalyClassification model wrapper
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

from typing import Any, Dict

import cv2
import numpy as np
from openvino.model_zoo.model_api.models import SegmentationModel
from openvino.model_zoo.model_api.models.types import NumericalValue


class AnomalyClassification(SegmentationModel):
    """
    Wrapper for anomaly classification task
    """

    __model__ = "anomaly_classification"

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters["resize_type"].update_default_value("crop")
        parameters.update(
            {
                "threshold": NumericalValue(default_value=0.2, description="Threshold value to locate anomaly"),
            }
        )

        return parameters

    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]) -> np.ndarray:
        outputs = outputs[self.output_blob_name].squeeze()
        input_image_height = meta["original_shape"][0]
        input_image_width = meta["original_shape"][1]
        meta["threshold"] = self.threshold  # pylint: disable=no-member

        result = cv2.resize(outputs, (input_image_width, input_image_height))
        return result
