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

import numpy as np
from openvino.model_api.adapters.inference_adapter import InferenceAdapter
from openvino.model_api.models import ImageModel, SegmentationModel
from openvino.model_api.models.types import NumericalValue, StringValue

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines import ResizeLongestSide


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
                "downsizing": NumericalValue(value_type=int, default_value=64, min=1, max=1024),
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

        self.mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        self.has_mask_input = np.zeros((1, 1), dtype=np.float32)

    @classmethod
    def parameters(cls):  # noqa: D102
        parameters = super().parameters()
        parameters.update({"image_size": NumericalValue(value_type=int, default_value=1024, min=0, max=2048)})
        parameters.update({"mask_threshold": NumericalValue(value_type=float, default_value=0.0, min=0, max=1)})
        return parameters

    def _get_outputs(self):
        return "upscaled_masks"

    def preprocess(self, inputs: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Preprocess prompts."""
        processed_prompts = []
        for prompt_name in ["bboxes", "points"]:
            for prompt, label in zip(inputs.get(prompt_name), inputs["labels"].get(prompt_name, [])):
                if prompt_name == "bboxes":
                    point_coords = self._apply_coords(prompt.reshape(-1, 2, 2), inputs["original_size"])
                    point_labels = np.array([2, 3], dtype=np.float32).reshape(-1, 2)
                else:
                    point_coords = self._apply_coords(prompt.reshape(-1, 1, 2), inputs["original_size"])
                    point_labels = np.array([1], dtype=np.float32).reshape(-1, 1)

                processed_prompts.append(
                    {
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                        "mask_input": self.mask_input,
                        "has_mask_input": self.has_mask_input,
                        "orig_size": np.asarray(inputs["original_size"], dtype=np.int64).reshape(-1, 2),
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
            soft_prediction (np.ndarray): The soft prediction.
        """
        probability = max(min(float(outputs["scores"]), 1.0), 0.0)
        hard_prediction = outputs[self.output_blob_name].squeeze() > self.mask_threshold
        soft_prediction = hard_prediction * probability

        meta["soft_prediction"] = soft_prediction
        meta["label"].probability = probability

        return hard_prediction, soft_prediction
