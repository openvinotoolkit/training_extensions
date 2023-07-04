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
from openvino.model_api.models import ClassificationModel, classification
from openvino.model_api.models.types import BooleanValue, DictValue


class OTXClassification(ClassificationModel):
    """OTX classification class for openvino."""

    __model__ = "otx_classification"

    def __init__(self, model_adapter, configuration=None, preload=False):
        backup_fn = classification.addOrFindSoftmaxAndTopkOutputs
        classification.addOrFindSoftmaxAndTopkOutputs = lambda a, b, c: None
        super().__init__(model_adapter, configuration, preload)
        classification.addOrFindSoftmaxAndTopkOutputs = backup_fn
        if self.hierarchical:
            logits_range_dict = self.multihead_class_info.get("head_idx_to_logits_range", False)
            if logits_range_dict:
                self.multihead_class_info["head_idx_to_logits_range"] = dict(logits_range_dict.items())
        self.out_layer_names = [self._get_output()]

    @classmethod
    def parameters(cls):
        """Parameters."""
        parameters = super().parameters()
        parameters["resize_type"].update_default_value("standard")
        parameters.update(
            {
                "multilabel": BooleanValue(default_value=False),
                "hierarchical": BooleanValue(default_value=False),
                "multihead_class_info": DictValue(default_value={}),
            }
        )

        return parameters

    def _check_io_number(self, number_of_inputs, number_of_outputs):
        pass

    def _get_output(self):
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

    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]):  # pylint: disable=unused-argument
        """Post-process."""
        logits = outputs[self.out_layer_names[-1]].squeeze()
        if self.multilabel:
            return get_multilabel_predictions(logits)
        if self.hierarchical:
            return get_hierarchical_predictions(logits, self.multihead_class_info)

        return get_multiclass_predictions(logits, activate=True)

    # pylint: disable=unused-argument
    def postprocess_aux_outputs(self, outputs: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        """Post-process for auxiliary outputs."""
        logits = outputs[self.out_layer_names[-1]].squeeze()
        if self.multilabel:
            probs = sigmoid_numpy(logits)
        elif self.hierarchical:
            probs = activate_multihead_output(logits, self.multihead_class_info)
        else:
            probs = softmax_numpy(logits)
        act_score = float(np.max(probs) - np.min(probs))

        if "saliency_map" in outputs:
            saliency_map = outputs["saliency_map"][0]
            repr_vector = outputs["feature_vector"].reshape(-1)
        else:
            saliency_map, repr_vector = None, None

        return probs, saliency_map, repr_vector, act_score


def sigmoid_numpy(x: np.ndarray):
    """Sigmoid numpy."""
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def softmax_numpy(x: np.ndarray, eps: float = 1e-9):
    """Softmax numpy."""
    x = np.exp(x - np.max(x))
    return x / (np.sum(x) + eps)


def activate_multihead_output(logits: np.ndarray, multihead_class_info: dict):
    """Activate multi-head output."""
    for i in range(multihead_class_info["num_multiclass_heads"]):
        logits_begin, logits_end = multihead_class_info["head_idx_to_logits_range"][str(i)]
        logits[logits_begin:logits_end] = softmax_numpy(logits[logits_begin:logits_end])

    if multihead_class_info["num_multilabel_classes"]:
        logits_begin = multihead_class_info["num_single_label_classes"]
        logits[logits_begin:] = sigmoid_numpy(logits[logits_begin:])

    return logits


def get_hierarchical_predictions(
    logits: np.ndarray, multihead_class_info: dict, pos_thr: float = 0.5, activate: bool = True
):
    """Get hierarchical predictions."""
    predicted_labels = []
    for i in range(multihead_class_info["num_multiclass_heads"]):
        logits_begin, logits_end = multihead_class_info["head_idx_to_logits_range"][str(i)]
        head_logits = logits[logits_begin:logits_end]
        if activate:
            head_logits = softmax_numpy(head_logits)
        j = np.argmax(head_logits)
        label_str = multihead_class_info["all_groups"][i][j]
        predicted_labels.append((multihead_class_info["label_to_idx"][label_str], head_logits[j]))

    if multihead_class_info["num_multilabel_classes"]:
        logits_begin = multihead_class_info["num_single_label_classes"]
        head_logits = logits[logits_begin:]
        if activate:
            head_logits = sigmoid_numpy(head_logits)

        for i in range(head_logits.shape[0]):
            if head_logits[i] > pos_thr:
                label_str = multihead_class_info["all_groups"][multihead_class_info["num_multiclass_heads"] + i][0]
                predicted_labels.append((multihead_class_info["label_to_idx"][label_str], head_logits[i]))

    return predicted_labels


def get_multiclass_predictions(logits: np.ndarray, activate: bool = True):
    """Get multiclass predictions."""
    index = np.argmax(logits)
    if activate:
        logits = softmax_numpy(logits)
    return [(index, logits[index])]


def get_multilabel_predictions(logits: np.ndarray, pos_thr: float = 0.5, activate: bool = True):
    """Get multilabel predictions."""
    if activate:
        logits = sigmoid_numpy(logits)
    scores = []
    indices = []
    for i in range(logits.shape[0]):
        if logits[i] > pos_thr:
            indices.append(i)
            scores.append(logits[i])

    return list(zip(indices, scores))
