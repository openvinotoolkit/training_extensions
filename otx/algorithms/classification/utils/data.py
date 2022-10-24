"""Data adapter from otx cli in Classifation Task."""

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

# pylint: disable=too-many-nested-blocks, invalid-name, too-many-locals

import json
import os
from enum import Enum, auto
from os import path as osp

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.utils.argument_checks import (
    OptionalDirectoryPathCheck,
    check_input_parameters_type,
)


class ClassificationType(Enum):
    """Classification Type."""

    MULTICLASS = auto()
    MULTILABEL = auto()
    MULTIHEAD = auto()


class ClassificationDatasetAdapter(DatasetEntity):
    """Classification Dataset Adapter from OTX CLI."""

    @check_input_parameters_type(
        {
            "train_ann_file": OptionalDirectoryPathCheck,
            "train_data_root": OptionalDirectoryPathCheck,
            "val_ann_file": OptionalDirectoryPathCheck,
            "val_data_root": OptionalDirectoryPathCheck,
            "test_ann_file": OptionalDirectoryPathCheck,
            "test_data_root": OptionalDirectoryPathCheck,
        }
    )
    def __init__(
        self,
        train_ann_file=None,
        train_data_root=None,
        val_ann_file=None,
        val_data_root=None,
        test_ann_file=None,
        test_data_root=None,
        **kwargs,
    ):
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
                if self.ann_files[k] and ".json" in self.ann_files[k] and osp.isfile(self.ann_files[k]):
                    self.annotations[k], self.data_type = self._load_text_annotation(
                        self.ann_files[k], self.data_roots[k]
                    )
                else:
                    self.annotations[k], self.data_type = self._load_annotation(self.data_roots[k])

        self.labels = None
        self._set_labels_obtained_from_annotation()
        self.project_labels = [
            LabelEntity(name=name, domain=Domain.CLASSIFICATION, is_empty=False, id=ID(i))
            for i, name in enumerate(self.labels)
        ]

        dataset_items = []
        for subset, subset_data in self.annotations.items():
            for data_info in subset_data[0]:
                image = Image(file_path=data_info[0])
                labels = [
                    ScoredLabel(label=self.label_name_to_project_label(label_name), probability=1.0)
                    for label_name in data_info[1]
                ]
                shapes = [Annotation(Rectangle.generate_full_box(), labels)]
                annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
                dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                dataset_items.append(dataset_item)

        super().__init__(items=dataset_items, **kwargs)

    @staticmethod
    def _load_text_annotation(annot_path, data_dir):
        out_data = []
        with open(annot_path, "rb") as f:
            annotation = json.load(f)
            if "hierarchy" not in annotation:
                all_classes = sorted(annotation["classes"])
                annotation_type = ClassificationType.MULTILABEL
                groups = [[c] for c in all_classes]
            else:  # load multihead
                all_classes = []
                groups = annotation["hierarchy"]

                def add_subtask_labels(group):
                    if isinstance(group, dict) and "subtask" in group:
                        subtask = group["subtask"]
                        if isinstance(subtask, list):
                            for task in subtask:
                                for task_label in task["labels"]:
                                    all_classes.append(task_label)
                        elif isinstance(subtask, dict):
                            for task_label in subtask["labels"]:
                                all_classes.append(task_label)
                        add_subtask_labels(subtask)
                    elif isinstance(group, list):
                        for task in group:
                            add_subtask_labels(task)

                for group in groups:
                    for label in group["labels"]:
                        all_classes.append(label)
                    add_subtask_labels(group)
                annotation_type = ClassificationType.MULTIHEAD

            images_info = annotation["images"]
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
                print(f"WARNING: there are {img_wo_objects} images without labels and will be treated as negatives")
        return (out_data, all_classes, groups), annotation_type

    @staticmethod
    def _load_annotation(data_dir, filter_classes=None):
        ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".gif")

        def is_valid(filename):
            return not filename.startswith(".") and filename.lower().endswith(ALLOWED_EXTS)

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
                        out_data.append((path, (target_class,), 0, 0, "", -1, -1))

        if not out_data:
            print("Failed to locate images in folder " + data_dir + f" with extensions {ALLOWED_EXTS}")

        all_classes = list(class_to_idx.keys())
        return (out_data, all_classes, [all_classes]), ClassificationType.MULTICLASS

    def _set_labels_obtained_from_annotation(self):
        self.labels = None
        for subset in self.data_roots:
            labels = self.annotations[subset][1]
            if self.labels and self.labels != labels:
                raise RuntimeError("Labels are different from annotation file to annotation file.")
            self.labels = labels
        assert self.labels is not None

    def label_name_to_project_label(self, label_name):
        """Return lists of project labels converted from label name."""
        return [label for label in self.project_labels if label.name == label_name][0]

    def is_multiclass(self):
        """Check if multi-class."""

        return self.data_type == ClassificationType.MULTICLASS

    def is_multilabel(self):
        """Check if multi-label."""
        return self.data_type == ClassificationType.MULTILABEL

    def is_multihead(self):
        """Check if multi-head."""
        return self.data_type == ClassificationType.MULTIHEAD

    def generate_label_schema(self):
        """Generate label schema."""
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
                    group_labels.append(self.label_name_to_project_label(cls))
                label_schema.add_group(
                    LabelGroup(name=group_labels[0].name, labels=group_labels, group_type=LabelGroupType.EXCLUSIVE)
                )
            label_schema.add_group(empty_group)
        return label_schema
