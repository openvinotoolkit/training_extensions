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
# pylint: disable=too-many-instance-attributes
class ClassificationDataset:
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
