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

from typing import Any, Dict, Iterable, Union

import cv2
import numpy as np

from otx.api.utils.argument_checks import check_input_parameters_type
import logging as log
try:
    from openvino.model_zoo.model_api.models.model import Model
    from openvino.model_zoo.model_api.models.types import BooleanValue, DictValue
    from openvino.model_zoo.model_api.models.utils import RESIZE_TYPES, pad_image, InputTransform
except ImportError as e:
    import warnings

    warnings.warn("ModelAPI was not found.")


class OTXActionCls(Model):
    """OTX classification class for openvino."""

    __model__ = "otx_action_classification"

    def __init__(self, model_adapter, configuration=None, preload=False):
        '''Image model constructor

        Calls the `Model` constructor first

        Args:
            model_adapter(ModelAdapter): allows working with the specified executor
            resize_type(str): sets the type for image resizing (see ``RESIZE_TYPE`` for info)
        '''
        super().__init__(model_adapter, configuration, preload)
        self.image_blob_names, self.image_info_blob_names = self._get_inputs()
        self.image_blob_name = self.image_blob_names[0]

        _, self.n, self.c, self.t, self.h, self.w = self.inputs[self.image_blob_name].shape
        self.resize = RESIZE_TYPES['standard']
        self.input_transform = InputTransform(False, None, None)

    @classmethod
    def parameters(cls):
        """Parameters."""
        parameters = super().parameters()
        # parameters["resize_type"].update_default_value("standard")
        return parameters

    def _check_io_number(self, number_of_inputs, number_of_outputs):
        pass

    def _get_inputs(self):
        image_blob_names, image_info_blob_names = [], []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 6:
                image_blob_names.append(name)
            elif len(metadata.shape) == 4:
                image_info_blob_names.append(name)
        #     else:
        #         raise RuntimeError('Failed to identify the input for ImageModel: only 2D and 4D input layer supported')
        # if not image_blob_names:
        #     raise RuntimeError('Failed to identify the input for the image: no 4D input layer found')
        return image_blob_names, image_info_blob_names

    def _get_outputs(self):
        layer_name = "logits"
        for name, meta in self.outputs.items():
            if "logits" in meta.names:
                layer_name = name
        layer_shape = self.outputs[layer_name].shape

        if len(layer_shape) != 2 and len(layer_shape) != 4:
            raise RuntimeError("The Classification model wrapper supports topologies only with 2D or 4D output")
        if len(layer_shape) == 4 and (layer_shape[2] != 1 or layer_shape[3] != 1):
            raise RuntimeError(
                "The Classification model wrapper supports topologies only with 4D "
                "output which has last two dimensions of size 1"
            )
        if self.labels:
            if layer_shape[1] == len(self.labels) + 1:
                self.labels.insert(0, "other")
                self.logger.warning("\tInserted 'other' label as first.")
            if layer_shape[1] != len(self.labels):
                raise RuntimeError(
                    "Model's number of classes and parsed "
                    f"labels must match ({layer_shape[1]} != {len(self.labels)})"
                )
        return layer_name

    # @check_input_parameters_type()
    # def preprocess(self, inputs: np.ndarray):
    #     """Pre-process."""
    #     meta = {"original_shape": inputs.shape}
    #     resized_image = self.resize(inputs, (self.w, self.h))
    #     resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    #     meta.update({"resized_shape": resized_image.shape})
    #     if self.resize_type == "fit_to_window":
    #         resized_image = pad_image(resized_image, (self.w, self.h))
    #     resized_image = self.input_transform(resized_image)
    #     resized_image = self._change_layout(resized_image)
    #     dict_inputs = {self.image_blob_name: resized_image}
    #     return dict_inputs, meta

    @check_input_parameters_type()
    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]):  # pylint: disable=unused-argument
        """Post-process."""
        logits = outputs[self.out_layer_name].squeeze()
        return get_multiclass_predictions(logits)

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
