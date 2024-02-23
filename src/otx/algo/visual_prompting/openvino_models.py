# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Openvino Model Wrappers for the OTX visual prompting."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from openvino.model_api.models import ImageModel, SegmentationModel
from openvino.model_api.models.types import BooleanValue, NumericalValue, StringValue

if TYPE_CHECKING:
    from openvino.model_api.adapters.inference_adapter import InferenceAdapter


class ImageEncoder(ImageModel):
    """Image Encoder class of OTX Visual Prompting model for openvino task."""

    __model__ = "image_encoder"

    def __init__(
        self,
        inference_adapter: InferenceAdapter,
        configuration: dict[str, Any] | None = None,
        preload: bool = False,
    ):
        super().__init__(inference_adapter, configuration, preload)

    @classmethod
    def parameters(cls) -> dict[str, Any]:  # noqa: D102
        parameters = super().parameters()
        parameters.update(
            {
                "resize_type": StringValue(default_value="fit_to_window"),
                "image_size": NumericalValue(value_type=int, default_value=1024, min=0, max=2048),
            },
        )
        return parameters

    def preprocess(self, inputs: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Update meta for image encoder."""
        dict_inputs, meta = super().preprocess(inputs)
        meta["resize_type"] = self.resize_type
        return dict_inputs, meta

    def postprocess(self, outputs: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, np.ndarray]:
        """Postprocess the model outputs."""
        return outputs


class PromptGetter(ImageModel):
    """Prompt Getter class of OTX Visual Prompting model for openvino task."""

    __model__ = "prompt_getter"

    def __init__(
        self,
        inference_adapter: InferenceAdapter,
        configuration: dict[str, Any] | None = None,
        preload: bool = False,
    ):
        super().__init__(inference_adapter, configuration, preload)

    @classmethod
    def parameters(cls) -> dict[str, Any]:  # noqa: D102
        parameters = super().parameters()
        parameters.update({"image_size": NumericalValue(value_type=int, default_value=1024, min=0, max=2048)})
        parameters.update({"sim_threshold": NumericalValue(value_type=float, default_value=0.5, min=0, max=1)})
        parameters.update({"num_bg_points": NumericalValue(value_type=int, default_value=1, min=0, max=1024)})
        parameters.update(
            {"default_threshold_reference": NumericalValue(value_type=float, default_value=0.3, min=-1.0, max=1.0)},
        )
        return parameters

    def _get_inputs(self) -> tuple[list[str], list[str]]:
        """Defines the model inputs for images and additional info."""
        image_blob_names, image_info_blob_names = [], []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 4:
                image_blob_names.append(name)
            else:
                image_info_blob_names.append(name)
        if not image_blob_names:
            self.raise_error("Failed to identify the input for the image: no 4D input layer found")
        return image_blob_names, image_info_blob_names


class Decoder(SegmentationModel):
    """Decoder class of OTX Visual Prompting model for openvino task."""

    __model__ = "decoder"

    def __init__(
        self,
        model_adapter: InferenceAdapter,
        configuration: dict | None = None,
        preload: bool = False,
    ):
        super().__init__(model_adapter, configuration, preload)

        self.mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        self.has_mask_input = np.zeros((1, 1), dtype=np.float32)

    @classmethod
    def parameters(cls) -> dict[str, Any]:  # noqa: D102
        parameters = super().parameters()
        parameters.update({"image_size": NumericalValue(value_type=int, default_value=1024, min=0, max=2048)})
        parameters.update({"mask_threshold": NumericalValue(value_type=float, default_value=0.0, min=0, max=1)})
        parameters.update({"embedded_processing": BooleanValue(default_value=True)})
        return parameters

    def _get_outputs(self) -> str:
        return "upscaled_masks"

    def preprocess(self, inputs: dict[str, Any]) -> list[dict[str, Any]]:
        """Preprocess prompts."""
        processed_prompts: list[dict[str, Any]] = []
        idx: int = 0
        for prompt_name in ["bboxes", "points"]:
            prompts = inputs.get(prompt_name, None)
            if prompts is None:
                continue
            for prompt in prompts:
                label = inputs["labels"][idx]
                if prompt_name == "bboxes":
                    point_coords = self._apply_coords(prompt.reshape(-1, 2, 2), inputs["orig_size"])
                    point_labels = np.array([2, 3], dtype=np.float32).reshape(-1, 2)
                else:
                    point_coords = self._apply_coords(prompt.reshape(-1, 1, 2), inputs["orig_size"])
                    point_labels = np.array([1], dtype=np.float32).reshape(-1, 1)

                processed_prompts.append(
                    {
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                        "mask_input": self.mask_input,
                        "has_mask_input": self.has_mask_input,
                        "orig_size": np.array(inputs["orig_size"], dtype=np.int64).reshape(-1, 2),
                        "label": label,
                    },
                )
                idx += 1
        return processed_prompts

    def _apply_coords(self, coords: np.ndarray, orig_size: list[int] | tuple[int, int]) -> np.ndarray:
        """Process coords according to preprocessed image size using image meta."""
        old_h, old_w = orig_size
        new_h, new_w = self._get_preprocess_shape(old_h, old_w, self.image_size)
        coords = deepcopy(coords).astype(np.float32)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def _get_preprocess_shape(self, old_h: int, old_w: int, image_size: int) -> tuple[int, int]:
        """Compute the output size given input size and target image size."""
        scale = image_size / max(old_h, old_w)
        new_h, new_w = old_h * scale, old_w * scale
        new_w = int(new_w + 0.5)
        new_h = int(new_h + 0.5)
        return (new_h, new_w)

    def _check_io_number(self, number_of_inputs: int | tuple[int], number_of_outputs: int | tuple[int]) -> None:
        pass

    def _get_inputs(self) -> tuple[list[str], list[str]]:
        """Get input layer name and shape."""
        image_blob_names = list(self.inputs.keys())
        image_info_blob_names: list = []
        return image_blob_names, image_info_blob_names

    def postprocess(self, outputs: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, np.ndarray]:
        """Postprocess to convert soft prediction to hard prediction.

        Args:
            outputs (dict[str, np.ndarray]): The output of the model.
            meta (dict[str, Any]): Contain label and original size.

        Returns:
            (dict[str, np.ndarray]): The postprocessed output of the model.
        """
        probability = max(min(float(outputs["scores"]), 1.0), 0.0)
        hard_prediction = outputs[self.output_blob_name].squeeze(1) > self.mask_threshold
        soft_prediction = hard_prediction * probability

        outputs["hard_prediction"] = hard_prediction
        outputs["soft_prediction"] = soft_prediction

        return outputs
