# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom OpenVINO model wrappers for video recognition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from openvino.model_api.adapters.utils import RESIZE_TYPES, InputTransform
from openvino.model_api.models.model import Model
from openvino.model_api.models.utils import (
    ClassificationResult,
)

if TYPE_CHECKING:
    from openvino.model_api.adapters import OpenvinoAdapter


def get_multiclass_predictions(logits: np.ndarray) -> ClassificationResult:
    """Get multiclass predictions."""
    index = np.argmax(logits)
    return ClassificationResult([(index, index, logits[index])], np.ndarray(0), np.ndarray(0), np.ndarray(0))


class OTXOVActionCls(Model):
    """OTX Action Classification model for openvino task."""

    __model__ = "Action Classification"

    def __init__(self, model_adapter: OpenvinoAdapter, configuration: dict | None = None, preload: bool = False):
        """Image model constructor.

        Calls the `Model` constructor first
        """
        super().__init__(model_adapter, configuration, preload)
        self.image_blob_names, self.image_info_blob_names = self._get_inputs()
        self.image_blob_name = self.image_blob_names[0]
        self.out_layer_name = self._get_outputs()

        _, self.n, self.c, self.t, self.h, self.w = self.inputs[self.image_blob_name].shape
        self.resize = RESIZE_TYPES["standard"]
        self.normalize = self._get_normalize_layer(model_adapter)
        self.input_transform = InputTransform(False, None, None)

        self.interval = 4

    def _get_inputs(self) -> tuple[list, list]:
        image_blob_names, image_info_blob_names = [], []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 6:
                image_blob_names.append(name)
            elif len(metadata.shape) == 4:
                image_info_blob_names.append(name)
        return image_blob_names, image_info_blob_names

    def _get_outputs(self) -> str:
        layer_name = "logits"
        for name, meta in self.outputs.items():
            if "logits" in meta.names:
                layer_name = name
        return layer_name

    @staticmethod
    def _get_normalize_layer(model_adapter: OpenvinoAdapter) -> Callable | None:
        model_info = None
        for key in model_adapter.model.rt_info:
            if key == "model_info":
                model_info = model_adapter.model.rt_info["model_info"]
        if model_info is None:
            return None
        scale_values = np.array(
            model_adapter.model.rt_info["model_info"]["scale_values"].value.split(),
            dtype=np.float64,
        )
        mean_values = np.array(
            model_adapter.model.rt_info["model_info"]["mean_values"].value.split(),
            dtype=np.float64,
        )
        return lambda x: (x - mean_values) / scale_values

    def preprocess(self, inputs: np.ndarray) -> tuple[dict, dict]:
        """Pre-process."""
        meta = {"original_shape": inputs[0].shape}
        frames = []
        for frame in inputs:
            resized_frame = self.resize(frame, (self.w, self.h))
            if self.normalize:
                resized_frame = self.normalize(resized_frame)
            frames.append(resized_frame)
        np_frames = self._reshape(frames)
        dict_inputs = {self.image_blob_name: np_frames}
        meta.update({"resized_shape": np_frames[0].shape})
        return dict_inputs, meta

    @staticmethod
    def _reshape(inputs: list[np.ndarray]) -> np.ndarray:
        """Reshape(expand, transpose, permute) the input np.ndarray."""
        np_inputs = np.expand_dims(inputs, axis=(0, 1))  # [1, 1, T, H, W, C]
        return np_inputs.transpose(0, 1, -1, 2, 3, 4)  # [1, 1, C, T, H, W]

    def postprocess(self, outputs: dict[str, np.ndarray], meta: dict[str, Any]) -> np.ndarray:
        """Post-process."""
        logits = next(iter(outputs.values())).squeeze()
        return get_multiclass_predictions(logits)
