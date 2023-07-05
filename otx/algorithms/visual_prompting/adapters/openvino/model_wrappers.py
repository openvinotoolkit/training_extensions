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
from openvino.model_zoo.model_api.models import ImageModel
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.utils.segmentation_utils import create_hard_prediction_from_soft_prediction


class ImageEncoder(ImageModel):
    """Image encoder class for visual prompting of openvino model wrapper."""
    __model__ = "image_encoder"

    @classmethod
    def parameters(cls) -> Dict[str, Any]:
        parameters = super().parameters()
        parameters["resize_type"].default_value = "fit_to_window"
        parameters["mean_values"].default_value = [123.675, 116.28, 103.53]
        parameters["scale_values"].default_value = [58.395, 57.12, 57.375]
        return parameters


class Decoder(BlurSegmentation):
    """Decoder class for visual prompting of openvino model wrapper.
    
    TODO (sungchul): change parent class
    """
    __model__ = "decoder"

    def preprocess(self, inputs: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """To be implemented."""
        return inputs

    def _get_inputs(self):
        """"""
        image_blob_names = [name for name in self.inputs.keys()]
        image_info_blob_names = []
        return image_blob_names, image_info_blob_names

    def _get_outputs(self):
        layer_name = "masks"
        layer_shape = self.outputs[layer_name].shape

        if len(layer_shape) == 3:
            self.out_channels = 0
        elif len(layer_shape) == 4:
            self.out_channels = layer_shape[1]
        else:
            raise Exception(f"Unexpected output layer shape {layer_shape}. Only 4D and 3D output layers are supported")

        return layer_name

    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Tuple[np.ndarray]:
        """"""
        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        soft_prediction = outputs[self.output_blob_name].squeeze()
        soft_prediction = sigmoid(soft_prediction)
        hard_prediction = create_hard_prediction_from_soft_prediction(
            soft_prediction=soft_prediction,
            soft_threshold=self.soft_threshold,
            blur_strength=self.blur_strength,
        )

        probability = max(min(float(outputs["iou_predictions"]), 1.0), 0.0)
        meta["label"].probability = probability

        return hard_prediction, soft_prediction
