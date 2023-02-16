"""Model Wrapper for blur of OTX Segmentation."""

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

import warnings
from typing import Any, Dict, Iterable, Optional, Union

import cv2
import numpy as np
from openvino.model_zoo.model_api.adapters.model_adapter import ModelAdapter
from openvino.model_zoo.model_api.models import SegmentationModel
from openvino.model_zoo.model_api.models.types import NumericalValue

from otx.api.utils.argument_checks import check_input_parameters_type
from otx.api.utils.segmentation_utils import create_hard_prediction_from_soft_prediction


@check_input_parameters_type()
def get_activation_map(features: Union[np.ndarray, Iterable, int, float]):
    """Getter activation_map functions."""
    min_soft_score = np.min(features)
    max_soft_score = np.max(features)
    factor = 255.0 / (max_soft_score - min_soft_score + 1e-12)

    float_act_map = factor * (features - min_soft_score)
    int_act_map = np.uint8(np.floor(float_act_map))
    int_act_map = cv2.applyColorMap(int_act_map, cv2.COLORMAP_JET)
    int_act_map = cv2.cvtColor(int_act_map, cv2.COLOR_BGR2RGB)
    return int_act_map


class BlurSegmentation(SegmentationModel):
    """BlurSegmentation class of openvino model wrapper."""

    __model__ = "blur_segmentation"

    @check_input_parameters_type()
    def __init__(self, model_adapter: ModelAdapter, configuration: Optional[dict] = None, preload: bool = False):
        super().__init__(model_adapter, configuration, preload)
        self.out_channels = 0

    @classmethod
    def parameters(cls):
        """BlurSegmentation.parameters function."""
        parameters = super().parameters()
        parameters.update(
            {
                "soft_threshold": NumericalValue(default_value=0.5, min=0.0, max=1.0),
                "blur_strength": NumericalValue(value_type=int, default_value=1, min=0, max=25),
            }
        )

        return parameters

    def _check_io_number(self, number_of_inputs, number_of_outputs):
        pass

    def _get_outputs(self):
        layer_name = "output"
        layer_shape = self.outputs[layer_name].shape

        if len(layer_shape) == 3:
            self.out_channels = 0
        elif len(layer_shape) == 4:
            self.out_channels = layer_shape[1]
        else:
            raise Exception(f"Unexpected output layer shape {layer_shape}. Only 4D and 3D output layers are supported")

        return layer_name

    @check_input_parameters_type()
    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]):
        """BlurSegmentation.postprocess function."""
        predictions = outputs[self.output_blob_name].squeeze()
        soft_prediction = np.transpose(predictions, axes=(1, 2, 0))

        hard_prediction = create_hard_prediction_from_soft_prediction(
            soft_prediction=soft_prediction, soft_threshold=self.soft_threshold, blur_strength=self.blur_strength
        )
        hard_prediction = cv2.resize(
            hard_prediction, meta["original_shape"][1::-1], 0, 0, interpolation=cv2.INTER_NEAREST
        )
        soft_prediction = cv2.resize(
            soft_prediction, meta["original_shape"][1::-1], 0, 0, interpolation=cv2.INTER_NEAREST
        )
        meta["soft_prediction"] = soft_prediction

        if "feature_vector" not in outputs:
            warnings.warn("Could not find Feature Vector in OpenVINO output. Please rerun export or retrain the model.")
            meta["feature_vector"] = None
        else:
            meta["feature_vector"] = outputs["feature_vector"].reshape(-1)

        return hard_prediction
