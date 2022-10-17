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

# pylint: disable=too-many-nested-blocks

from contextlib import contextmanager
from enum import Enum, auto
import importlib
import json
import math
import os
import shutil
import tempfile
import time
from os import path as osp
from operator import itemgetter
from typing import Iterable, List, Union

import numpy as np

from otx.api.entities.datasets import DatasetEntity

from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import (LabelGroup, LabelGroupType,
                                           LabelSchemaEntity)
from otx.api.entities.model_template import ModelTemplate
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.train_parameters import UpdateProgressCallback
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from torch.nn.modules import Module

from torchreid.utils import set_model_attr, get_model_attr

@check_input_parameters_type()
def generate_label_schema(not_empty_labels: List[LabelEntity], multilabel: bool = False):
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


@check_input_parameters_type()
def get_multihead_class_info(label_schema: LabelSchemaEntity):
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
    for i, g in enumerate(exclusive_groups):
        head_idx_to_logits_range[i] = (last_logits_pos, last_logits_pos + len(g))
        last_logits_pos += len(g)
        for j, c in enumerate(g):
            class_to_idx[c] = (i, j)  # group idx and idx inside group
            num_single_label_classes += 1

    # other labels are in multilabel group
    for j, g in enumerate(single_label_groups):
        class_to_idx[g[0]] = (len(exclusive_groups), j)

    all_labels = label_schema.get_labels(include_empty=False)
    label_to_idx = {lbl.name: i for i, lbl in enumerate(all_labels)}

    mixed_cls_heads_info = {
                            'num_multiclass_heads': len(exclusive_groups),
                            'num_multilabel_classes': len(single_label_groups),
                            'head_idx_to_logits_range': head_idx_to_logits_range,
                            'num_single_label_classes': num_single_label_classes,
                            'class_to_group_idx': class_to_idx,
                            'all_groups': exclusive_groups + single_label_groups,
                            'label_to_idx': label_to_idx
                            }
    return mixed_cls_heads_info


class OTEClassificationDataset:
    @check_input_parameters_type({"ote_dataset": DatasetParamTypeCheck})
    def __init__(self, ote_dataset: DatasetEntity, labels: List[LabelEntity], multilabel: bool = False,
                 hierarchical: bool = False, mixed_cls_heads_info: dict = {}, keep_empty_label: bool = False):
        super().__init__()
        self.ote_dataset = ote_dataset
        self.multilabel = multilabel
        self.mixed_cls_heads_info = mixed_cls_heads_info
        self.hierarchical = hierarchical
        self.labels = labels
        self.annotation = []
        self.keep_empty_label = keep_empty_label
        self.label_names = [label.name for label in self.labels]

        for i, _ in enumerate(self.ote_dataset):
            class_indices = []
            item_labels = self.ote_dataset[i].get_roi_labels(self.labels,
                                                             include_empty=self.keep_empty_label)
            ignored_labels = self.ote_dataset[i].ignored_labels
            if item_labels:
                if not self.hierarchical:
                    for ote_lbl in item_labels:
                        if ote_lbl not in ignored_labels:
                            class_indices.append(self.label_names.index(ote_lbl.name))
                        else:
                            class_indices.append(-1)
                else:
                    num_cls_heads = self.mixed_cls_heads_info['num_multiclass_heads']

                    class_indices = [0]*(self.mixed_cls_heads_info['num_multiclass_heads'] + \
                                         self.mixed_cls_heads_info['num_multilabel_classes'])
                    for j in range(num_cls_heads):
                        class_indices[j] = -1
                    for ote_lbl in item_labels:
                        group_idx, in_group_idx = self.mixed_cls_heads_info['class_to_group_idx'][ote_lbl.name]
                        if group_idx < num_cls_heads:
                            class_indices[group_idx] = in_group_idx
                        else:
                            if ote_lbl not in ignored_labels:
                                class_indices[num_cls_heads + in_group_idx] = 1
                            else:
                                class_indices[num_cls_heads + in_group_idx] = -1

            else:  # this supposed to happen only on inference stage or if we have a negative in multilabel data
                if self.mixed_cls_heads_info:
                    class_indices = [-1]*(self.mixed_cls_heads_info['num_multiclass_heads'] + \
                                          self.mixed_cls_heads_info['num_multilabel_classes'])
                else:
                    class_indices.append(-1)

            if self.multilabel or self.hierarchical:
                self.annotation.append({'label': tuple(class_indices)})
            else:
                self.annotation.append({'label': class_indices[0]})

    @check_input_parameters_type()
    def __getitem__(self, idx: int):
        sample = self.ote_dataset[idx].numpy  # This returns 8-bit numpy array of shape (height, width, RGB)
        label = self.annotation[idx]['label']
        return {'img': sample, 'label': label}

    def __len__(self):
        return len(self.annotation)

    def get_annotation(self):
        return self.annotation

    def get_classes(self):
        return self.label_names


@check_input_parameters_type()
def get_task_class(path: str):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@check_input_parameters_type()
def reload_hyper_parameters(model_template: ModelTemplate):
    """ This function copies template.yaml file and its configuration.yaml dependency to temporal folder.
        Then it re-loads hyper parameters from copied template.yaml file.
        This function should not be used in general case, it is assumed that
        the 'configuration.yaml' should be in the same folder as 'template.yaml' file.
    """

    template_file = model_template.model_template_path
    template_dir = osp.dirname(template_file)
    temp_folder = tempfile.mkdtemp()
    conf_yaml = [dep.source for dep in model_template.dependencies \
                     if dep.destination == model_template.hyper_parameters.base_path][0]
    conf_yaml = osp.join(template_dir, conf_yaml)
    shutil.copy(conf_yaml, temp_folder)
    shutil.copy(template_file, temp_folder)
    model_template.hyper_parameters.load_parameters(osp.join(temp_folder, 'template_experimental.yaml'))
    assert model_template.hyper_parameters.data


@check_input_parameters_type()
def set_values_as_default(parameters: dict):
    for v in parameters.values():
        if isinstance(v, dict) and 'value' not in v:
            set_values_as_default(v)
        elif isinstance(v, dict) and 'value' in v:
            if v['value'] != v['default_value']:
                v['value'] = v['default_value']


@contextmanager
@check_input_parameters_type()
def force_fp32(model: Module):
    mix_precision_status = get_model_attr(model, 'mix_precision')
    set_model_attr(model, 'mix_precision', False)
    try:
        yield model
    finally:
        set_model_attr(model, 'mix_precision', mix_precision_status)


class InferenceProgressCallback(TimeMonitorCallback):
    def __init__(self, num_test_steps, update_progress_callback: UpdateProgressCallback):
        super().__init__(
            num_epoch=0,
            num_train_steps=0,
            num_val_steps=0,
            num_test_steps=num_test_steps,
            update_progress_callback=update_progress_callback)

    def on_test_batch_end(self, batch=None, logs=None):
        super().on_test_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress())


class OptimizationProgressCallback(TimeMonitorCallback):
    """ Progress callback used for optimization using NNCF
        There are three stages to the progress bar:
           - 5 % model is loaded
           - 10 % compressed model is initialized
           - 10-100 % compressed model is being fine-tuned
    """
    def __init__(self, update_progress_callback: UpdateProgressCallback, loading_stage_progress_percentage: int = 5,
                 initialization_stage_progress_percentage: int = 5, **kwargs):
        super().__init__(update_progress_callback=update_progress_callback, **kwargs)
        if loading_stage_progress_percentage + initialization_stage_progress_percentage >= 100:
            raise RuntimeError('Total optimization progress percentage is more than 100%')

        train_percentage = 100 - loading_stage_progress_percentage - initialization_stage_progress_percentage
        self.loading_stage_steps = self.total_steps * loading_stage_progress_percentage / train_percentage
        self.initialization_stage_steps = self.total_steps * initialization_stage_progress_percentage / train_percentage
        self.total_steps += self.loading_stage_steps + self.initialization_stage_steps

        # set loading_stage_steps from the start as the model is already loaded at this point
        self.current_step = self.loading_stage_steps
        self.update_progress_callback(self.get_progress())

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress(), score=logs)

    def on_train_end(self, logs=None):
        super(OptimizationProgressCallback, self).on_train_end(logs)
        self.update_progress_callback(self.get_progress(), score=logs)

    def on_initialization_end(self):
        self.current_step += self.initialization_stage_steps
        self.update_progress_callback(self.get_progress())


@check_input_parameters_type()
def active_score_from_probs(predictions: Union[np.ndarray, Iterable, int, float]):
    top_idxs = np.argpartition(predictions, -2)[-2:]
    top_probs = predictions[top_idxs]
    return np.max(top_probs) - np.min(top_probs)


@check_input_parameters_type()
def sigmoid_numpy(x: np.ndarray):
    return 1. / (1. + np.exp(-1. * x))


@check_input_parameters_type()
def softmax_numpy(x: np.ndarray):
    x = np.exp(x - np.max(x))
    x /= np.sum(x)
    return x


@check_input_parameters_type()
def get_multiclass_predictions(logits: np.ndarray, labels: List[LabelEntity],
                               activate: bool = True) -> List[ScoredLabel]:
    i = np.argmax(logits)
    if activate:
        logits = softmax_numpy(logits)
    if math.isnan(float(logits[i])):
        return []
    return [ScoredLabel(labels[i], probability=float(logits[i]))]


@check_input_parameters_type()
def get_multilabel_predictions(logits: np.ndarray, labels: List[LabelEntity],
                               pos_thr: float = 0.5, activate: bool = True) -> List[ScoredLabel]:
    if activate:
        logits = sigmoid_numpy(logits)
    item_labels = []
    for i in range(logits.shape[0]):
        if logits[i] > pos_thr:
            label = ScoredLabel(label=labels[i], probability=float(logits[i]))
            item_labels.append(label)

    return item_labels


@check_input_parameters_type()
def get_hierarchical_predictions(logits: np.ndarray, labels: List[LabelEntity],
                                 label_schema: LabelSchemaEntity, multihead_class_info: dict,
                                 pos_thr: float = 0.5, activate: bool = True) -> List[ScoredLabel]:
    predicted_labels = []
    for i in range(multihead_class_info['num_multiclass_heads']):
        logits_begin, logits_end = multihead_class_info['head_idx_to_logits_range'][i]
        head_logits = logits[logits_begin : logits_end]
        if activate:
            head_logits = softmax_numpy(head_logits)
        j = np.argmax(head_logits)
        label_str = multihead_class_info['all_groups'][i][j]
        ote_label = next(x for x in labels if x.name == label_str)
        predicted_labels.append(ScoredLabel(label=ote_label, probability=float(head_logits[j])))

    if multihead_class_info['num_multilabel_classes']:
        logits_begin, logits_end = multihead_class_info['num_single_label_classes'], -1
        head_logits = logits[logits_begin : logits_end]
        if activate:
            head_logits = sigmoid_numpy(head_logits)

        for i in range(head_logits.shape[0]):
            if head_logits[i] > pos_thr:
                label_str = multihead_class_info['all_groups'][multihead_class_info['num_multiclass_heads'] + i][0]
                ote_label = next(x for x in labels if x.name == label_str)
                predicted_labels.append(ScoredLabel(label=ote_label, probability=float(head_logits[i])))

    return label_schema.resolve_labels_probabilistic(predicted_labels)
