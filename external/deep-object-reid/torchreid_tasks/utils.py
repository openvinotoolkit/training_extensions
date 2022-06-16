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
from typing import List

import cv2 as cv
import numpy as np

from ote_sdk.entities.annotation import (Annotation, AnnotationSceneEntity,
                                         AnnotationSceneKind)
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import (LabelGroup, LabelGroupType,
                                           LabelSchemaEntity)
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.train_parameters import UpdateProgressCallback
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback

from torchreid.utils import set_model_attr, get_model_attr

class ClassificationType(Enum):
    MULTICLASS = auto()
    MULTILABEL = auto()
    MULTIHEAD = auto()


class ClassificationDatasetAdapter(DatasetEntity):
    def __init__(self,
                 train_ann_file=None,
                 train_data_root=None,
                 val_ann_file=None,
                 val_data_root=None,
                 test_ann_file=None,
                 test_data_root=None,
                 **kwargs):
        self.data_roots = {}
        self.ann_files = {}
        self.data_type = ClassificationType.MULTICLASS
        if train_data_root:
            self.data_roots[Subset.TRAINING] = train_data_root
            self.ann_files[Subset.TRAINING] = train_ann_file
        if val_data_root:
            self.data_roots[Subset.VALIDATION] = val_data_root
            self.ann_files[Subset.VALIDATION] = val_ann_file
        if test_data_root:
            self.data_roots[Subset.TESTING] = test_data_root
            self.ann_files[Subset.TESTING] = test_ann_file
        self.annotations = {}
        for k, v in self.data_roots.items():
            if v:
                self.data_roots[k] = osp.abspath(v)
                if self.ann_files[k] and '.json' in self.ann_files[k] and osp.isfile(self.ann_files[k]):
                    self.data_roots[k] = osp.dirname(self.ann_files[k])
                    self.annotations[k], self.data_type = \
                        self._load_text_annotation(self.ann_files[k], self.data_roots[k])
                else:
                    self.annotations[k], self.data_type = self._load_annotation(self.data_roots[k])

        self.labels = None
        self._set_labels_obtained_from_annotation()
        self.project_labels = [LabelEntity(name=name, domain=Domain.CLASSIFICATION,
                                           is_empty=False, id=ID(i)) for i, name in enumerate(self.labels)]

        dataset_items = []
        for subset, subset_data in self.annotations.items():
            for data_info in subset_data[0]:
                image = Image(file_path=data_info[0])
                labels = [ScoredLabel(label=self._label_name_to_project_label(label_name),
                                      probability=1.0) for label_name in data_info[1]]
                shapes = [Annotation(Rectangle.generate_full_box(), labels)]
                annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION,
                                                         annotations=shapes)
                dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                dataset_items.append(dataset_item)

        super().__init__(items=dataset_items, **kwargs)

    @staticmethod
    def _load_text_annotation(annot_path, data_dir):
        out_data = []
        with open(annot_path) as f:
            annotation = json.load(f)
            if not 'label_groups' in annotation:
                all_classes = sorted(annotation['classes'])
                annotation_type = ClassificationType.MULTILABEL
                groups = [[c] for c in all_classes]
            else: # load multihead
                groups = annotation['label_groups']
                all_classes = []
                for g in groups:
                    for c in g:
                        all_classes.append(c)
                annotation_type = ClassificationType.MULTIHEAD

            images_info = annotation['images']
            img_wo_objects = 0
            for img_info in images_info:
                rel_image_path, img_labels = img_info
                full_image_path = osp.join(data_dir, rel_image_path)
                labels_idx = [lbl for lbl in img_labels if lbl in all_classes]
                assert full_image_path
                if not labels_idx:
                    img_wo_objects += 1
                out_data.append((full_image_path, tuple(labels_idx)))
            if img_wo_objects:
                print(f'WARNING: there are {img_wo_objects} images without labels and will be treated as negatives')
        return (out_data, all_classes, groups), annotation_type

    @staticmethod
    def _load_annotation(data_dir, filter_classes=None):
        ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.gif')
        def is_valid(filename):
            return not filename.startswith('.') and filename.lower().endswith(ALLOWED_EXTS)

        def find_classes(folder, filter_names=None):
            if filter_names:
                classes = [d.name for d in os.scandir(folder) if d.is_dir() and d.name in filter_names]
            else:
                classes = [d.name for d in os.scandir(folder) if d.is_dir()]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return class_to_idx

        class_to_idx = find_classes(data_dir, filter_classes)

        out_data = []
        for target_class in sorted(class_to_idx.keys()):
            # class_index = class_to_idx[target_class]
            target_dir = osp.join(data_dir, target_class)
            if not osp.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = osp.join(root, fname)
                    if is_valid(path):
                        out_data.append((path, (target_class, ), 0, 0, '', -1, -1))

        if not out_data:
            print('Failed to locate images in folder ' + data_dir + f' with extensions {ALLOWED_EXTS}')

        all_classes = list(class_to_idx.keys())
        return (out_data, all_classes, [all_classes]), ClassificationType.MULTICLASS

    def _set_labels_obtained_from_annotation(self):
        self.labels = None
        for subset in self.data_roots:
            labels = self.annotations[subset][1]
            if self.labels and self.labels != labels:
                raise RuntimeError('Labels are different from annotation file to annotation file.')
            self.labels = labels
        assert self.labels is not None

    def _label_name_to_project_label(self, label_name):
        return [label for label in self.project_labels if label.name == label_name][0]

    def is_multilabel(self):
        return self.data_type == ClassificationType.MULTILABEL

    def generate_label_schema(self):
        label_schema = LabelSchemaEntity()
        if self.data_type == ClassificationType.MULTICLASS:
            main_group = LabelGroup(name="labels", labels=self.project_labels, group_type=LabelGroupType.EXCLUSIVE)
            label_schema.add_group(main_group)
        elif self.data_type in [ClassificationType.MULTIHEAD, ClassificationType.MULTILABEL]:
            emptylabel = LabelEntity(name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION)
            empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
            for g in self.annotations[Subset.TRAINING][2]:
                group_labels = []
                for cls in g:
                    group_labels.append(self._label_name_to_project_label(cls))
                label_schema.add_group(LabelGroup(name=group_labels[0].name,
                                                  labels=group_labels, group_type=LabelGroupType.EXCLUSIVE))
            label_schema.add_group(empty_group)
        return label_schema


def generate_label_schema(not_empty_labels, multilabel=False):
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
            class_to_idx[c] = (i, j) # group idx and idx inside group
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
    def __init__(self, ote_dataset: DatasetEntity, labels, multilabel=False, hierarchical=False,
                 mixed_cls_heads_info={}, keep_empty_label=False):
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
                        if not ote_lbl in ignored_labels:
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
                            if not ote_lbl in ignored_labels:
                                class_indices[num_cls_heads + in_group_idx] = 1
                            else:
                                class_indices[num_cls_heads + in_group_idx] = -1

            else: # this supposed to happen only on inference stage or if we have a negative in multilabel data
                if self.mixed_cls_heads_info:
                    class_indices = [-1]*(self.mixed_cls_heads_info['num_multiclass_heads'] + \
                                          self.mixed_cls_heads_info['num_multilabel_classes'])
                else:
                    class_indices.append(-1)

            if self.multilabel or self.hierarchical:
                self.annotation.append({'label': tuple(class_indices)})
            else:
                self.annotation.append({'label': class_indices[0]})

    def __getitem__(self, idx):
        sample = self.ote_dataset[idx].numpy  # This returns 8-bit numpy array of shape (height, width, RGB)
        label = self.annotation[idx]['label']
        return {'img': sample, 'label': label}

    def __len__(self):
        return len(self.annotation)

    def get_annotation(self):
        return self.annotation

    def get_classes(self):
        return self.label_names


def get_task_class(path):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def reload_hyper_parameters(model_template):
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
    model_template.hyper_parameters.load_parameters(osp.join(temp_folder, 'template.yaml'))
    assert model_template.hyper_parameters.data


def set_values_as_default(parameters):
    for v in parameters.values():
        if isinstance(v, dict) and 'value' not in v:
            set_values_as_default(v)
        elif isinstance(v, dict) and 'value' in v:
            if v['value'] != v['default_value']:
                v['value'] = v['default_value']


@contextmanager
def force_fp32(model):
    mix_precision_status = get_model_attr(model, 'mix_precision')
    set_model_attr(model, 'mix_precision', False)
    try:
        yield model
    finally:
        set_model_attr(model, 'mix_precision', mix_precision_status)


class TrainingProgressCallback(TimeMonitorCallback):
    def __init__(self, update_progress_callback: UpdateProgressCallback, **kwargs):
        super().__init__(update_progress_callback=update_progress_callback, **kwargs)

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress(), score=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.past_epoch_duration.append(time.time() - self.start_epoch_time)
        self._calculate_average_epoch()
        self.update_progress_callback(self.get_progress(), score=float(logs))


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


def preprocess_features_for_actmap(features):
    features = np.mean(features, axis=1)
    b, h, w = features.shape
    features = features.reshape(b, h * w)
    features = np.exp(features)
    features /= np.sum(features, axis=1, keepdims=True)
    features = features.reshape(b, h, w)
    return features


def get_actmap(features, output_res):
    am = cv.resize(features, output_res)
    am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
    am = np.uint8(np.floor(am))
    am = cv.applyColorMap(am, cv.COLORMAP_JET)
    return am


def active_score_from_probs(predictions):
    top_idxs = np.argpartition(predictions, -2)[-2:]
    top_probs = predictions[top_idxs]
    return np.max(top_probs) - np.min(top_probs)


def sigmoid_numpy(x: np.ndarray):
    return 1. / (1. + np.exp(-1. * x))


def softmax_numpy(x: np.ndarray):
    x = np.exp(x - np.max(x))
    x /= np.sum(x)
    return x


def get_multiclass_predictions(logits: np.ndarray, labels: List[LabelEntity],
                               activate: bool = True) -> List[ScoredLabel]:
    i = np.argmax(logits)
    if activate:
        logits = softmax_numpy(logits)
    if math.isnan(float(logits[i])):
        return []
    return [ScoredLabel(labels[i], probability=float(logits[i]))]


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
