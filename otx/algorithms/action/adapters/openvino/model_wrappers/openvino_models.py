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

import glob
from typing import Any, Dict

import cv2
import numpy as np

from otx.api.entities.datasets import DatasetItemEntity
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

    warnings.warn("ModelAPI was not found.")


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
    def preprocess(self, inputs: DatasetItemEntity):
        """Pre-process."""
        frames = []
        # TODO: allow only .jpg, .png exts
        rawframes = glob.glob(inputs.media["frame_dir"] + "/*")  # type: ignore[index]
        rawframes.sort()
        for rawframe in rawframes:
            frame = cv2.imread(rawframe)
            resized_frame = self.resize(frame, (self.w, self.h))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
            resized_frame = self.input_transform(resized_frame)
            frames.append(resized_frame)
        np_frames = np.expand_dims(frames, axis=(0, 1))  # [1, 1, T, H, W, C]
        np_frames = np_frames.transpose(0, 1, -1, 2, 3, 4)  # [1, 1, C, T, H, W]
        frame_inds = self.get_frame_inds(np_frames)
        np_frames = np_frames[:, :, :, frame_inds, :, :]
        dict_inputs = {self.image_blob_name: np_frames}
        meta = {"original_shape": np_frames.shape}
        meta.update({"resized_shape": resized_frame.shape})
        return dict_inputs, meta

    def get_frame_inds(self, np_frames: np.ndarray):
        """Get sampled index for given video input."""
        frame_len = np_frames.shape[3]
        ori_clip_len = self.t * self.interval
        if frame_len > ori_clip_len - 1:
            start = (frame_len - ori_clip_len + 1) / 2
        else:
            start = 0
        frame_inds = np.arange(self.t) * self.interval + int(start)
        frame_inds = np.clip(frame_inds, 0, frame_len - 1)
        frame_inds = frame_inds.astype(np.int)
        return frame_inds

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
    def preprocess(self, inputs: DatasetItemEntity):
        """Pre-process."""
        # SampleFrame -> RawFrameDecode -> Resize
        frames = []
        rawframes = glob.glob(inputs.media["frame_dir"] + "/*")  # type: ignore[index]
        rawframes.sort()
        for rawframe in rawframes:
            frame = cv2.imread(rawframe)
            resized_frame = self.resize(frame, (self.w, self.h))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
            resized_frame = self.input_transform(resized_frame)
            frames.append(resized_frame)
        np_frames = np.expand_dims(frames, axis=(0))  # [1, T, H, W, C]
        np_frames = np_frames.transpose(0, 4, 1, 2, 3)  # [1, C, T, H, W]
        frame_inds = self.get_frame_inds(np_frames)
        np_frames = np_frames[:, :, frame_inds, :, :]
        dict_inputs = {self.image_blob_name: np_frames}
        meta = {"original_shape": frame.shape}
        meta.update({"resized_shape": resized_frame.shape})
        return dict_inputs, meta

    def get_frame_inds(self, np_frames: np.ndarray):
        """Get sampled index for given np_frames.

        Sample clips from middle frame of video
        """
        timestamp = np_frames.shape[2] // 2
        timestamp_start = 1
        timestamp_end = np_frames.shape[2]
        shot_info = (0, timestamp_end - timestamp_start) * self.fps

        center_index = self.fps * (timestamp - timestamp_start) + 1

        start = center_index - (self.t // 2) * self.frame_interval
        end = center_index + ((self.t + 1) // 2) * self.frame_interval
        frame_inds = list(range(start, end, self.frame_interval))
        frame_inds = np.clip(frame_inds, shot_info[0], shot_info[1] - 1)
        return frame_inds

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
