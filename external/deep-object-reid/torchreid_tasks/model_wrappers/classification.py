# Copyright (C) 2021 Intel Corporation
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

import cv2
import numpy as np
from typing import Any, Dict

try:
    from openvino.model_zoo.model_api.models.classification import Classification
    from openvino.model_zoo.model_api.models.types import BooleanValue
    from openvino.model_zoo.model_api.models.utils import pad_image
except ImportError as e:
    import warnings
    warnings.warn("ModelAPI was not found.")


class OteClassification(Classification):
    __model__ = 'ote_classification'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters['resize_type'].update_default_value('standard')
        parameters.update({
            'multilabel': BooleanValue(default_value=False)
        })

        return parameters

    def _check_io_number(self, inp, outp):
        pass

    def _get_outputs(self):
        layer_name = 'logits'
        layer_shape = self.outputs[layer_name].shape

        if len(layer_shape) != 2 and len(layer_shape) != 4:
            raise RuntimeError('The Classification model wrapper supports topologies only with 2D or 4D output')
        if len(layer_shape) == 4 and (layer_shape[2] != 1 or layer_shape[3] != 1):
            raise RuntimeError('The Classification model wrapper supports topologies only with 4D '
                               'output which has last two dimensions of size 1')
        if self.labels:
            if (layer_shape[1] == len(self.labels) + 1):
                self.labels.insert(0, 'other')
                self.logger.warning("\tInserted 'other' label as first.")
            if layer_shape[1] != len(self.labels):
                raise RuntimeError("Model's number of classes and parsed "
                                'labels must match ({} != {})'.format(layer_shape[1], len(self.labels)))
        return layer_name

    def preprocess(self, image: np.ndarray):
        meta = {'original_shape': image.shape}
        resized_image = self.resize(image, (self.w, self.h))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        meta.update({'resized_shape': resized_image.shape})
        if self.resize_type == 'fit_to_window':
            resized_image = pad_image(resized_image, (self.w, self.h))
        resized_image = self.input_transform(resized_image)
        resized_image = self._change_layout(resized_image)
        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        logits = outputs[self.out_layer_name].squeeze()
        if self.multilabel:
            predictions = get_multilabel_predictions(logits)

        predictions = get_multiclass_predictions(logits)

        return predictions

    def postprocess_aux_outputs(self, outputs: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        features = preprocess_features_for_actmap(outputs['features'])
        actmap = get_actmap(features[0], (metadata['original_shape'][1], metadata['original_shape'][0]))
        repr_vector = outputs['vector']

        logits = outputs[self.out_layer_name].squeeze()
        if self.multilabel:
            probs = sigmoid_numpy(logits)
        probs = softmax_numpy(logits)

        act_score = float(np.max(probs) - np.min(probs))

        return actmap, repr_vector, act_score



def preprocess_features_for_actmap(features):
    features = np.mean(features, axis=1)
    b, h, w = features.shape
    features = features.reshape(b, h * w)
    features = np.exp(features)
    features /= np.sum(features, axis=1, keepdims=True)
    features = features.reshape(b, h, w)
    return features


def get_actmap(features, output_res):
    am = cv2.resize(features, output_res)
    am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
    am = np.uint8(np.floor(am))
    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
    return am


def sigmoid_numpy(x: np.ndarray):
    return 1. / (1. + np.exp(-1. * x))


def softmax_numpy(x: np.ndarray):
    x = np.exp(x)
    x /= np.sum(x)
    return x


def get_multiclass_predictions(logits: np.ndarray, activate: bool = True):

    index = np.argmax(logits)
    if activate:
        logits = softmax_numpy(logits)
    return [(index, logits[index])]


def get_multilabel_predictions(logits: np.ndarray, pos_thr: float = 0.5, activate: bool = True):
    if activate:
        logits = sigmoid_numpy(logits)
    scores = []
    indices = []
    for i in range(logits.shape[0]):
        if logits[i] > pos_thr:
            indices.append(i)
            scores.append(logits[i])

    return list(zip(indices, scores))
