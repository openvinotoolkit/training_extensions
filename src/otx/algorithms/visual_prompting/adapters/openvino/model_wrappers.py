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

import cv2
import numpy as np
from openvino.model_api.models import ImageModel
from openvino.model_api.models.types import NumericalValue

from otx.algorithms.segmentation.adapters.openvino.model_wrappers.blur import (
    BlurSegmentation,
)
from otx.api.utils.segmentation_utils import create_hard_prediction_from_soft_prediction


class ImageEncoder(ImageModel):
    """Image encoder class for visual prompting of openvino model wrapper."""

    __model__ = "image_encoder"

    @classmethod
    def parameters(cls) -> Dict[str, Any]:  # noqa: D102
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

    def preprocess(self, bbox: np.ndarray, original_size: Tuple[int]) -> Dict[str, Any]:
        """Ready decoder inputs."""
        point_coords = bbox.reshape((-1, 2, 2))
        point_labels = np.array([2, 3], dtype=np.float32).reshape((-1, 2))
        inputs_decoder = {
            "point_coords": point_coords,
            "point_labels": point_labels,
            # TODO (sungchul): how to generate mask_input and has_mask_input
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros((1, 1), dtype=np.float32),
            "orig_size": np.array(original_size, dtype=np.float32).reshape((-1, 2)),
        }
        return inputs_decoder

    @classmethod
    def parameters(cls):  # noqa: D102
        parameters = super().parameters()
        parameters.update({"image_size": NumericalValue(value_type=int, default_value=1024, min=0, max=2048)})
        return parameters

    def _get_inputs(self):
        """Get input layer name and shape."""
        image_blob_names = [name for name in self.inputs.keys()]
        image_info_blob_names = []
        return image_blob_names, image_info_blob_names

    def _get_outputs(self):
        """Get output layer name and shape."""
        layer_name = "low_res_masks"
        layer_shape = self.outputs[layer_name].shape

        if len(layer_shape) == 3:
            self.out_channels = 0
        elif len(layer_shape) == 4:
            self.out_channels = layer_shape[1]
        else:
            raise Exception(f"Unexpected output layer shape {layer_shape}. Only 4D and 3D output layers are supported")

        return layer_name

    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess to convert soft prediction to hard prediction.

        Args:
            outputs (Dict[str, np.ndarray]): The output of the model.
            meta (Dict[str, Any]): Contain label and original size.

        Returns:
            hard_prediction (np.ndarray): The hard prediction.
            soft_prediction (np.ndarray): Resized, cropped, and normalized soft prediction.
        """

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        soft_prediction = outputs[self.output_blob_name].squeeze()
        soft_prediction = self.resize_and_crop(soft_prediction, meta["original_size"])
        soft_prediction = sigmoid(soft_prediction)
        meta["soft_prediction"] = soft_prediction

        hard_prediction = create_hard_prediction_from_soft_prediction(
            soft_prediction=soft_prediction,
            soft_threshold=self.soft_threshold,
            blur_strength=self.blur_strength,
        )

        probability = max(min(float(outputs["iou_predictions"]), 1.0), 0.0)
        meta["label"].probability = probability

        return hard_prediction, soft_prediction

    def resize_and_crop(self, soft_prediction: np.ndarray, original_size: np.ndarray) -> np.ndarray:
        """Resize and crop soft prediction.

        Args:
            soft_prediction (np.ndarray): Predicted soft prediction with HxW shape.
            original_size (np.ndarray): The original image size.

        Returns:
            final_soft_prediction (np.ndarray): Resized and cropped soft prediction for the original image.
        """
        resized_soft_prediction = cv2.resize(
            soft_prediction, (self.image_size, self.image_size), 0, 0, interpolation=cv2.INTER_LINEAR
        )

        prepadded_size = self.resize_longest_image_size(original_size, self.image_size).astype(np.int64)
        resized_cropped_soft_prediction = resized_soft_prediction[..., : prepadded_size[0], : prepadded_size[1]]

        original_size = original_size.astype(np.int64)
        h, w = original_size[0], original_size[1]
        final_soft_prediction = cv2.resize(
            resized_cropped_soft_prediction, (w, h), 0, 0, interpolation=cv2.INTER_LINEAR
        )
        return final_soft_prediction

    def resize_longest_image_size(self, original_size: np.ndarray, longest_side: int) -> np.ndarray:
        """Resizes the longest side of the image to the given size.

        Args:
            original_size (np.ndarray): The original image size with shape Bx2.
            longest_side (int): The size of the longest side.

        Returns:
            transformed_size (np.ndarray): The transformed image size with shape Bx2.
        """
        original_size = original_size.astype(np.float32)
        scale = longest_side / np.max(original_size)
        transformed_size = scale * original_size
        transformed_size = np.floor(transformed_size + 0.5).astype(np.int64)
        return transformed_size
