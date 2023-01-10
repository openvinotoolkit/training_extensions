"""Model wrapper file for openvino."""

# Copyright (C) 2022 Intel Corporation
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

# pylint: disable=invalid-name

from typing import Any, Dict

import numpy as np

from otx.api.entities.datasets import DatasetEntity
from otx.api.utils.argument_checks import check_input_parameters_type

try:
    from openvino.model_zoo.model_api.models.model import Model
    from openvino.model_zoo.model_api.models.utils import (
        RESIZE_TYPES,
        Detection,
        InputTransform,
    )
except ImportError as e:
    import warnings

    warnings.warn(f"{e}, ModelAPI was not found.")


@check_input_parameters_type()
def softmax_numpy(x: np.ndarray):
    """Softmax numpy."""
    x = np.exp(x)
    x /= np.sum(x)
    return x


@check_input_parameters_type()
def get_multiclass_predictions(logits: np.ndarray, activate: bool = True):
    """Get multiclass predictions."""
    index = np.argmax(logits)
    if activate:
        logits = softmax_numpy(logits)
    return [(index, logits[index])]


# pylint: disable=too-many-instance-attributes
class OTXActionCls(Model):
    """OTX Action Classification model for openvino task."""

    __model__ = "ACTION_CLASSIFICATION"

    def __init__(self, model_adapter, configuration=None, preload=False):
        """Image model constructor.

        Calls the `Model` constructor first
        """
        super().__init__(model_adapter, configuration, preload)
        self.image_blob_names, self.image_info_blob_names = self._get_inputs()
        self.image_blob_name = self.image_blob_names[0]
        self.out_layer_name = self._get_outputs()

        _, self.n, self.c, self.t, self.h, self.w = self.inputs[self.image_blob_name].shape
        self.resize = RESIZE_TYPES["standard"]
        self.input_transform = InputTransform(False, None, None)

        # FIXME Below parameters should be changed dynamically from data pipeline config
        self.interval = 4

    def _get_inputs(self):
        image_blob_names, image_info_blob_names = [], []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 6:
                image_blob_names.append(name)
            elif len(metadata.shape) == 4:
                image_info_blob_names.append(name)
        return image_blob_names, image_info_blob_names

    def _get_outputs(self):
        layer_name = "logits"
        for name, meta in self.outputs.items():
            if "logits" in meta.names:
                layer_name = name
        return layer_name

    @check_input_parameters_type()
    def preprocess(self, idx: int, dataset: DatasetEntity):
        """Pre-process."""
        start = idx - (self.t // 2) * self.interval
        end = idx + ((self.t + 1)) // 2 * self.interval
        frame_inds = list(range(start, end, self.interval))
        frame_inds = np.clip(frame_inds, 0, len(dataset) - 1)
        frames = []
        meta = {"original_shape": dataset[0].media.numpy.shape}
        for index in frame_inds:
            dataset_item = dataset[int(index)]
            frame = dataset_item.media.numpy
            frame = self.resize(frame, (self.w, self.h))
            frames.append(frame)
        np_frames = np.expand_dims(frames, axis=(0, 1))  # [1, 1, T, H, W, C]
        np_frames = np_frames.transpose(0, 1, -1, 2, 3, 4)  # [1, 1, C, T, H, W]
        dict_inputs = {self.image_blob_name: np_frames}
        meta.update({"resized_shape": np_frames[0].shape})
        return dict_inputs, meta

    @check_input_parameters_type()
    # pylint: disable=unused-argument
    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]):
        """Post-process."""
        logits = outputs[self.out_layer_name].squeeze()
        return get_multiclass_predictions(logits)


class OTXActionDet(OTXActionCls):
    """OTX Action Detection model for openvino task."""

    __model__ = "ACTION_DETECTION"

    def __init__(self, model_adapter, configuration=None, preload=False):
        """Image model constructor.

        Calls the `Model` constructor first
        """
        super(OTXActionCls, self).__init__(model_adapter, configuration, preload)
        self.image_blob_names = self._get_inputs()
        self.image_blob_name = self.image_blob_names[0]
        self.out_layer_names = self._get_outputs()

        self.n, self.c, self.t, self.h, self.w = self.inputs[self.image_blob_name].shape
        self.resize = RESIZE_TYPES["standard"]
        self.input_transform = InputTransform(False, None, None)

        # FIXME Below parameters should be changed dynamically from data pipeline config
        self.interval = 1
        self.fps = 1

    def _get_inputs(self):
        image_blob_names = []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 5:
                image_blob_names.append(name)
        return image_blob_names

    def _get_outputs(self):
        out_names = {}
        for name in self.outputs:
            if "bboxes" in name:
                out_names["bboxes"] = name
            elif "labels" in name:
                out_names["labels"] = name
        return out_names

    @check_input_parameters_type()
    def preprocess(self, idx: int, dataset: DatasetEntity):
        """Pre-process."""
        start = idx - (self.t // 2) * self.interval
        end = idx + ((self.t + 1)) // 2 * self.interval
        frame_inds = list(range(start, end, self.interval))
        frame_inds = np.clip(frame_inds, 0, len(dataset) - 1)
        frames = []
        meta = {"original_shape": dataset[0].media.numpy.shape}
        for index in frame_inds:
            dataset_item = dataset[int(index)]
            frame = dataset_item.media.numpy
            frame = self.resize(frame, (self.w, self.h))
            frames.append(frame)
        np_frames = np.expand_dims(frames, axis=(0))  # [1, T, H, W, C]
        np_frames = np_frames.transpose(0, -1, 1, 2, 3)  # [1, C, T, H, W]
        dict_inputs = {self.image_blob_name: np_frames}
        meta.update({"resized_shape": np_frames[0].shape})
        return dict_inputs, meta

    @check_input_parameters_type()
    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]):
        """Post-process."""
        # TODO Support multi class classification
        H, W, _ = meta["original_shape"]
        bboxes = outputs[self.out_layer_names["bboxes"]]
        labels = outputs[self.out_layer_names["labels"]]
        scores = labels[:, 1:].max(axis=1)
        labels = labels[:, 1:].argmax(axis=1)
        results = []
        for bbox, score, label in zip(bboxes, scores, labels):
            bbox *= [W, H, W, H]
            results.append(Detection(*bbox, score, label))
        return results
