# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Union, Iterable

import cv2
import numpy as np

from mpa.utils.logger import get_logger
from ote_sdk.entities.datasets import DatasetEntity

logger = get_logger()


def get_cls_img_indices(labels, dataset):
    img_indices = {label.name: list() for label in labels}
    for i, item in enumerate(dataset):
        item_labels = item.annotation_scene.get_labels()
        for i_l in item_labels:
            if i_l in labels:
                img_indices[i_l.name].append(i)

    return img_indices


def get_old_new_img_indices(labels, new_classes, dataset):
    ids_old, ids_new = [], []
    _dataset_label_schema_map = {label.name: label for label in labels}
    new_classes = [_dataset_label_schema_map[new_class] for new_class in new_classes]
    for i, item in enumerate(dataset):
        if item.annotation_scene.contains_any(new_classes):
            ids_new.append(i)
        else:
            ids_old.append(i)
    return {'old': ids_old, 'new': ids_new}


def get_actmap(saliency_map: Union[np.ndarray, Iterable, int, float],
               output_res: Union[tuple, list]):
    saliency_map = cv2.resize(saliency_map, output_res)
    saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
    return saliency_map

def convert_to_one_hot_indice(labels, class_indices):
    onehot_indices = np.zeros(len(labels))
    for idx in class_indices:
        if idx != -1:  # TODO: handling ignored label?
            onehot_indices[idx] = 1
    return onehot_indices

def convert_to_mmcls_dataset(gt_dataset: DatasetEntity, labels: list, include_empty=False, multiclass=False):
    gt_labels = []
    label_names = [label.name for label in labels]
    for gt_item in gt_dataset:
        class_indices = []
        item_labels = gt_item.get_roi_labels(labels, include_empty=include_empty)
        ignored_labels = gt_item.ignored_labels
        if item_labels:
            for ote_lbl in item_labels:
                if ote_lbl not in ignored_labels:
                    class_indices.append(label_names.index(ote_lbl.name))
                else:
                    class_indices.append(-1)
        else:  # this supposed to happen only on inference stage or if we have a negative in multilabel data
            class_indices.append(-1)
        if multiclass is True:
            gt_label = convert_to_one_hot_indice(labels, class_indices)
        else:
            gt_label = class_indices
        gt_labels.append(gt_label)
    gt_labels = np.array(gt_labels)
    return gt_labels, label_names