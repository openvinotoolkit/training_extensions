"""Utils for deep_object_reid tasks."""

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

# pylint: disable=too-many-nested-blocks, invalid-name

import math
import shutil
import tempfile
from contextlib import contextmanager
from operator import itemgetter
from os import path as osp
from typing import List

import numpy as np
from torch.nn.modules import Module
from torchreid.utils import get_model_attr, set_model_attr

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model_template import ModelTemplate
from otx.api.entities.scored_label import ScoredLabel
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)


@check_input_parameters_type()
def generate_label_schema(not_empty_labels: List[LabelEntity], multilabel: bool = False):
    """Generate label schema."""
    assert len(not_empty_labels) > 1

    label_schema = LabelSchemaEntity()
    if multilabel:
        emptylabel = LabelEntity(name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION)
        empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
        for label in not_empty_labels:
            label_schema.add_group(LabelGroup(name=label.name, labels=[label], group_type=LabelGroupType.EXCLUSIVE))
        label_schema.add_group(empty_group)
    else:
        main_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
        label_schema.add_group(main_group)
    return label_schema


# pylint: disable=invalid-name, too-many-locals
@check_input_parameters_type()
def get_multihead_class_info(label_schema: LabelSchemaEntity):
    """Get multihead class info by label schema."""
    all_groups = label_schema.get_groups(include_empty=False)
    all_groups_str = []
    for g in all_groups:
        group_labels_str = [lbl.name for lbl in g.labels]
        all_groups_str.append(group_labels_str)

    single_label_groups = [g for g in all_groups_str if len(g) == 1]
    exclusive_groups = [sorted(g) for g in all_groups_str if len(g) > 1]
    single_label_groups.sort(key=itemgetter(0))
    exclusive_groups.sort(key=itemgetter(0))
    class_to_idx = {}
    head_idx_to_logits_range = {}
    num_single_label_classes = 0
    last_logits_pos = 0
    for i, group in enumerate(exclusive_groups):
        head_idx_to_logits_range[i] = (last_logits_pos, last_logits_pos + len(group))
        last_logits_pos += len(group)
        for j, cls in enumerate(group):
            class_to_idx[cls] = (i, j)  # group idx and idx inside group
            num_single_label_classes += 1

    # other labels are in multilabel group
    for j, group in enumerate(single_label_groups):
        class_to_idx[group[0]] = (len(exclusive_groups), j)

    all_labels = label_schema.get_labels(include_empty=False)
    label_to_idx = {lbl.name: i for i, lbl in enumerate(all_labels)}

    mixed_cls_heads_info = {
        "num_multiclass_heads": len(exclusive_groups),
        "num_multilabel_classes": len(single_label_groups),
        "head_idx_to_logits_range": head_idx_to_logits_range,
        "num_single_label_classes": num_single_label_classes,
        "class_to_group_idx": class_to_idx,
        "all_groups": exclusive_groups + single_label_groups,
        "label_to_idx": label_to_idx,
    }
    return mixed_cls_heads_info


# pylint: disable=too-many-instance-attributes
class DORClassificationDataset:
    """Dataset used in deep_object_reid tasks."""

    @check_input_parameters_type({"otx_dataset": DatasetParamTypeCheck})
    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: List[LabelEntity],
        multilabel: bool = False,
        hierarchical: bool = False,
        mixed_cls_heads_info: dict = None,
        keep_empty_label: bool = False,
    ):  # pylint: disable=too-many-branches, too-many-locals
        super().__init__()
        self.otx_dataset = otx_dataset
        self.multilabel = multilabel
        self.mixed_cls_heads_info = mixed_cls_heads_info
        self.hierarchical = hierarchical
        self.labels = labels
        self.annotation = []
        self.keep_empty_label = keep_empty_label
        self.label_names = [label.name for label in self.labels]

        for i, _ in enumerate(self.otx_dataset):
            class_indices = []
            item_labels = self.otx_dataset[i].get_roi_labels(self.labels, include_empty=self.keep_empty_label)
            ignored_labels = self.otx_dataset[i].ignored_labels
            if item_labels:
                if not self.hierarchical:
                    for otx_lbl in item_labels:
                        if otx_lbl not in ignored_labels:
                            class_indices.append(self.label_names.index(otx_lbl.name))
                        else:
                            class_indices.append(-1)
                else:
                    if self.mixed_cls_heads_info is None:
                        raise TypeError("mixed_cls_heads_info is NoneType.")
                    num_cls_heads = self.mixed_cls_heads_info["num_multiclass_heads"]

                    class_indices = [0] * (
                        self.mixed_cls_heads_info["num_multiclass_heads"]
                        + self.mixed_cls_heads_info["num_multilabel_classes"]
                    )
                    for j in range(num_cls_heads):
                        class_indices[j] = -1
                    for otx_lbl in item_labels:
                        group_idx, in_group_idx = self.mixed_cls_heads_info["class_to_group_idx"][otx_lbl.name]
                        if group_idx < num_cls_heads:
                            class_indices[group_idx] = in_group_idx
                        else:
                            if otx_lbl not in ignored_labels:
                                class_indices[num_cls_heads + in_group_idx] = 1
                            else:
                                class_indices[num_cls_heads + in_group_idx] = -1

            else:  # this supposed to happen only on inference stage or if we have a negative in multilabel data
                if self.mixed_cls_heads_info:
                    class_indices = [-1] * (
                        self.mixed_cls_heads_info["num_multiclass_heads"]
                        + self.mixed_cls_heads_info["num_multilabel_classes"]
                    )
                else:
                    class_indices.append(-1)

            if self.multilabel or self.hierarchical:
                self.annotation.append({"label": tuple(class_indices)})
            else:
                self.annotation.append({"label": class_indices[0]})  # type: ignore

    @check_input_parameters_type()
    def __getitem__(self, idx: int):
        """Get item from dataset."""
        sample = self.otx_dataset[idx].numpy  # This returns 8-bit numpy array of shape (height, width, RGB)
        label = self.annotation[idx]["label"]
        return {"img": sample, "label": label}

    def __len__(self):
        """Get annotation length."""
        return len(self.annotation)

    def get_annotation(self):
        """Get annotation."""
        return self.annotation

    def get_classes(self):
        """Get classes' name."""
        return self.label_names


@check_input_parameters_type()
def reload_hyper_parameters(model_template: ModelTemplate):
    """Reload hyper-parameters function.

    This function copies template.yaml file and its configuration.yaml dependency to temporal folder.
    Then it re-loads hyper parameters from copied template.yaml file.
    This function should not be used in general case, it is assumed that
    the 'configuration.yaml' should be in the same folder as 'template.yaml' file.
    """

    template_file = model_template.model_template_path
    template_dir = osp.dirname(template_file)
    temp_folder = tempfile.mkdtemp()
    conf_yaml = [
        dep.source
        for dep in model_template.dependencies
        if dep.destination == model_template.hyper_parameters.base_path
    ][0]
    conf_yaml = osp.join(template_dir, conf_yaml)
    shutil.copy(conf_yaml, temp_folder)
    shutil.copy(template_file, temp_folder)
    model_template.hyper_parameters.load_parameters(osp.join(temp_folder, "template_experimental.yaml"))
    assert model_template.hyper_parameters.data


@check_input_parameters_type()
def set_values_as_default(parameters: dict):
    """Set values as default."""
    for v in parameters.values():
        if isinstance(v, dict) and "value" not in v:
            set_values_as_default(v)
        elif isinstance(v, dict) and "value" in v:
            if v["value"] != v["default_value"]:
                v["value"] = v["default_value"]


@contextmanager
@check_input_parameters_type()
def force_fp32(model: Module):
    """Force fp32."""
    mix_precision_status = get_model_attr(model, "mix_precision")
    set_model_attr(model, "mix_precision", False)
    try:
        yield model
    finally:
        set_model_attr(model, "mix_precision", mix_precision_status)


class InferenceProgressCallback(TimeMonitorCallback):
    """Progress callback used for inference.

    There are three stages to the progress bar:
       - 5 % model is loaded
       - 10 % compressed model is initialized
       - 10-100 % compressed model is being fine-tuned
    """

    def __init__(self, num_test_steps, update_progress_callback):
        super().__init__(
            num_epoch=0,
            num_train_steps=0,
            num_val_steps=0,
            num_test_steps=num_test_steps,
            update_progress_callback=update_progress_callback,
        )

    def on_test_batch_end(self, batch=None, logs=None):
        """Callback when batch-test ended."""
        super().on_test_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress())


class OptimizationProgressCallback(TimeMonitorCallback):
    """Progress callback used for optimization using NNCF.

    There are three stages to the progress bar:
       - 5 % model is loaded
       - 10 % compressed model is initialized
       - 10-100 % compressed model is being fine-tuned
    """

    def __init__(
        self,
        update_progress_callback,
        loading_stage_progress_percentage: int = 5,
        initialization_stage_progress_percentage: int = 5,
        **kwargs
    ):
        super().__init__(update_progress_callback=update_progress_callback, **kwargs)
        if loading_stage_progress_percentage + initialization_stage_progress_percentage >= 100:
            raise RuntimeError("Total optimization progress percentage is more than 100%")

        train_percentage = 100 - loading_stage_progress_percentage - initialization_stage_progress_percentage
        self.loading_stage_steps = self.total_steps * loading_stage_progress_percentage / train_percentage
        self.initialization_stage_steps = self.total_steps * initialization_stage_progress_percentage / train_percentage
        self.total_steps += self.loading_stage_steps + self.initialization_stage_steps  # type: ignore

        # set loading_stage_steps from the start as the model is already loaded at this point
        self.current_step = self.loading_stage_steps  # type: ignore
        self.update_progress_callback(self.get_progress())  # type: ignore

    def on_train_batch_end(self, batch, logs=None):
        """Callback when batch-train ended."""
        super().on_train_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress(), score=logs)

    def on_train_end(self, logs=None):
        """Callback when train ended."""
        super().on_train_end(logs)
        self.update_progress_callback(self.get_progress(), score=logs)

    def on_initialization_end(self):
        """Callback when init ended."""
        self.current_step += self.initialization_stage_steps
        self.update_progress_callback(self.get_progress())


@check_input_parameters_type()
def active_score_from_probs(predictions):
    """Active score form probs."""
    top_idxs = np.argpartition(predictions, -2)[-2:]
    top_probs = predictions[top_idxs]
    return np.max(top_probs) - np.min(top_probs)


@check_input_parameters_type()
def sigmoid_numpy(x: np.ndarray):
    """Sigmoid numpy."""
    return 1.0 / (1.0 + np.exp(-1.0 * x))


@check_input_parameters_type()
def softmax_numpy(x: np.ndarray):
    """Softmax numpy."""
    x = np.exp(x - np.max(x))
    x /= np.sum(x)
    return x


@check_input_parameters_type()
def get_multiclass_predictions(
    logits: np.ndarray, labels: List[LabelEntity], activate: bool = True
) -> List[ScoredLabel]:
    """Get multiclass predictions."""
    i = np.argmax(logits)
    if activate:
        logits = softmax_numpy(logits)
    if math.isnan(float(logits[i])):
        return []
    return [ScoredLabel(labels[i], probability=float(logits[i]))]


@check_input_parameters_type()
def get_multilabel_predictions(
    logits: np.ndarray, labels: List[LabelEntity], pos_thr: float = 0.5, activate: bool = True
) -> List[ScoredLabel]:
    """Get multilabel predictions."""
    if activate:
        logits = sigmoid_numpy(logits)
    item_labels = []
    for i in range(logits.shape[0]):
        if logits[i] > pos_thr:
            label = ScoredLabel(label=labels[i], probability=float(logits[i]))
            item_labels.append(label)

    return item_labels


@check_input_parameters_type()
def get_hierarchical_predictions(
    logits: np.ndarray,
    labels: List[LabelEntity],
    label_schema: LabelSchemaEntity,
    multihead_class_info: dict,
    pos_thr: float = 0.5,
    activate: bool = True,
) -> List[ScoredLabel]:
    """Get hierarchical predictions."""
    predicted_labels = []
    for i in range(multihead_class_info["num_multiclass_heads"]):
        logits_begin, logits_end = multihead_class_info["head_idx_to_logits_range"][i]
        head_logits = logits[logits_begin:logits_end]
        if activate:
            head_logits = softmax_numpy(head_logits)
        j = np.argmax(head_logits)
        label_str = multihead_class_info["all_groups"][i][j]
        otx_label = next(x for x in labels if x.name == label_str)
        predicted_labels.append(ScoredLabel(label=otx_label, probability=float(head_logits[j])))

    if multihead_class_info["num_multilabel_classes"]:
        logits_begin, logits_end = multihead_class_info["num_single_label_classes"], -1
        head_logits = logits[logits_begin:logits_end]
        if activate:
            head_logits = sigmoid_numpy(head_logits)

        for i in range(head_logits.shape[0]):
            if head_logits[i] > pos_thr:
                label_str = multihead_class_info["all_groups"][multihead_class_info["num_multiclass_heads"] + i][0]
                otx_label = next(x for x in labels if x.name == label_str)
                predicted_labels.append(ScoredLabel(label=otx_label, probability=float(head_logits[i])))

    return label_schema.resolve_labels_probabilistic(predicted_labels)
