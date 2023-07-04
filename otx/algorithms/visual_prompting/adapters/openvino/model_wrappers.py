"""Model Wrapper of OTX Visual Prompting."""

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

from typing import Any, Dict, Tuple

import numpy as np
from openvino.model_zoo.model_api.models.image_model import StringValue
from openvino.model_zoo.model_api.models.utils import RESIZE_TYPES

from otx.algorithms.segmentation.adapters.openvino.model_wrappers.blur import (
    BlurSegmentation,
)
from otx.api.entities.dataset_item import DatasetItemEntity


class VisualPrompting(BlurSegmentation):
    """VisualPrompting class of openvino model wrapper.
    
    This class inherits from BlurSegmentation to postprocess predictions with the same way.
    """

    __model__ = "visual_prompting"

    def preprocess_prompt(self, mask: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        pass

    @classmethod
    def parameters(cls) -> Dict[str, Any]:
        parameters = super().parameters()
        parameters["resize_type"].default_value = "fit_to_window"
        parameters["mean_values"].default_value = [123.675, 116.28, 103.53]
        parameters["scale_values"].default_value = [58.395, 57.12, 57.375]
        return parameters

    def _get_inputs(self):
        """Get input names for image encoder and decoder.

        Return:
            image_blob_names (list): List of image encoder's inputs.
            image_info_blob_names (list): List of decoder's inputs.
        """
        image_blob_names = ["images"]
        image_info_blob_names = [name for name in self.inputs["decoder"].keys()]
        if not image_blob_names:
            self.raise_error('Failed to identify the input for the image: no 4D input layer found')
        return image_blob_names, image_info_blob_names

    def _get_outputs(self):
        """Get output names for image encoder and decoder."""
        return {module: list(meta.keys()) for module, meta in self.outputs.items()}
