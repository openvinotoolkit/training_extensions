"""Openvino Model Wrappers of OTX Visual Prompting."""

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

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from openvino.model_api.adapters.inference_adapter import InferenceAdapter
from openvino.model_api.models import ImageModel, SegmentationModel
from openvino.model_api.models.types import NumericalValue, StringValue

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines import ResizeLongestSide
from otx.api.utils.segmentation_utils import create_hard_prediction_from_soft_prediction


class ImageEncoder(ImageModel):
    """Image encoder class for visual prompting of openvino model wrapper."""

    __model__ = "image_encoder"

    def __init__(self, inference_adapter, configuration=None, preload=False):
        super().__init__(inference_adapter, configuration, preload)

    @classmethod
    def parameters(cls) -> Dict[str, Any]:  # noqa: D102
        parameters = super().parameters()
        parameters.update(
            {
                "resize_type": StringValue(default_value="fit_to_window"),
                "image_size": NumericalValue(value_type=int, default_value=1024, min=0, max=2048),
            }
        )
        return parameters

    def preprocess(
        self, inputs: np.ndarray, extra_processing: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Update meta for image encoder."""
        dict_inputs, meta = super().preprocess(inputs)
        if extra_processing:
            dict_inputs["images"] = ResizeLongestSide.apply_image(dict_inputs["images"][0], self.image_size).transpose(
                2, 0, 1
            )[None]
        meta["resize_type"] = self.resize_type
        return dict_inputs, meta


class Decoder(SegmentationModel):
    """Decoder class for visual prompting of openvino model wrapper."""

    __model__ = "decoder"

    def __init__(
        self,
        model_adapter: InferenceAdapter,
        configuration: Optional[dict] = None,
        preload: bool = False,
    ):
        super().__init__(model_adapter, configuration, preload)

    @classmethod
    def parameters(cls):  # noqa: D102
        parameters = super().parameters()
        parameters.update({"image_size": NumericalValue(value_type=int, default_value=1024, min=0, max=2048)})
        return parameters

    def _get_outputs(self):
        return "low_res_masks"

    def preprocess(self, inputs: Dict[str, Any], meta: Dict[str, Any]):
        """Preprocess prompts."""
        processed_prompts = []
        # TODO (sungchul): process points
        for bbox, label in zip(inputs["bboxes"], inputs["labels"]):
            # TODO (sungchul): add condition to check whether using bbox or point
            point_coords = self._apply_coords(bbox.reshape(-1, 2, 2), inputs["original_size"])
            point_labels = np.array([2, 3], dtype=np.float32).reshape((-1, 2))
            processed_prompts.append(
                {
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    # TODO (sungchul): how to generate mask_input and has_mask_input
                    "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                    "has_mask_input": np.zeros((1, 1), dtype=np.float32),
                    "orig_size": np.array(inputs["original_size"], dtype=np.float32).reshape((-1, 2)),
                    "label": label,
                }
            )
        return processed_prompts

    def _apply_coords(self, coords: np.ndarray, original_size: Union[List[int], Tuple[int, int]]) -> np.ndarray:
        """Process coords according to preprocessed image size using image meta."""
        old_h, old_w = original_size
        new_h, new_w = self._get_preprocess_shape(original_size[0], original_size[1], self.image_size)
        coords = deepcopy(coords).astype(np.float32)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def _get_preprocess_shape(self, old_h: int, old_w: int, image_size: int) -> Tuple[int, int]:
        """Compute the output size given input size and target image size."""
        scale = image_size / max(old_h, old_w)
        new_h, new_w = old_h * scale, old_w * scale
        new_w = int(new_w + 0.5)
        new_h = int(new_h + 0.5)
        return (new_h, new_w)

    def _check_io_number(self, number_of_inputs, number_of_outputs):
        pass

    def _get_inputs(self):
        """Get input layer name and shape."""
        image_blob_names = [name for name in self.inputs.keys()]
        image_info_blob_names = []
        return image_blob_names, image_info_blob_names

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
            return np.tanh(x * 0.5) * 0.5 + 0.5  # to avoid overflow

        soft_prediction = outputs[self.output_blob_name].squeeze()
        soft_prediction = self.resize_and_crop(soft_prediction, meta["original_size"][0])
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

        prepadded_size = self.get_padded_size(original_size, self.image_size).astype(np.int64)
        resized_cropped_soft_prediction = resized_soft_prediction[..., : prepadded_size[0], : prepadded_size[1]]

        original_size = original_size.astype(np.int64)
        h, w = original_size
        final_soft_prediction = cv2.resize(
            resized_cropped_soft_prediction, (w, h), 0, 0, interpolation=cv2.INTER_LINEAR
        )
        return final_soft_prediction

    def get_padded_size(self, original_size: np.ndarray, longest_side: int) -> np.ndarray:
        """Get padded size from original size and longest side of the image.

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
