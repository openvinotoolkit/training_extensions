import os
from copy import deepcopy

import numpy as np
from numpy.lib.function_base import select
from sc_sdk.entities.annotation import (Annotation, AnnotationKind,
                                        NullMediaIdentifier)
from sc_sdk.entities.datasets import Dataset, DatasetItem, NullDataset, Subset
from sc_sdk.entities.image import Image
from sc_sdk.entities.label import Label, ScoredLabel
from sc_sdk.entities.shapes.box import Box

from .coco import CocoDataset, get_classes_from_annotation


class ObjectDetectionDataset(Dataset):

    def __init__(self,
                 train_ann_file=None,
                 train_data_root=None,
                 val_ann_file=None,
                 val_data_root=None,
                 test_ann_file=None,
                 test_data_root=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.ann_files = {}
        self.data_roots = {}
        self.ann_files[Subset.TRAINING] = train_ann_file
        self.data_roots[Subset.TRAINING] = train_data_root
        self.ann_files[Subset.VALIDATION] = val_ann_file
        self.data_roots[Subset.VALIDATION] = val_data_root
        self.ann_files[Subset.TESTING] = test_ann_file
        self.data_roots[Subset.TESTING] = test_data_root
        self.coco_dataset = None

        for k, v in self.ann_files.items():
            if v:
                self.ann_files[k] = os.path.abspath(v)

        for k, v in self.data_roots.items():
            if v:
                self.data_roots[k] = os.path.abspath(v)

        self.set_labels_obtained_from_annotation()

    def set_labels_obtained_from_annotation(self):
        self.labels = None
        for subset in (Subset.TRAINING, Subset.VALIDATION, Subset.TESTING):
            path = self.ann_files[subset]
            if path:
                labels = get_classes_from_annotation(path)
                if self.labels and self.labels != labels:
                    raise RuntimeError('Labels are different from annotation file to annotation file.')
                self.labels = labels
        assert self.labels is not None

    def set_project_labels(self, proejct_labels):
        self.project_labels = proejct_labels

    def label_name_to_project_label(self, label_name):
        return [label for label in self.project_labels if label.name == label_name][0]

    def init_as_subset(self, subset: Subset):
        test_mode = subset in {Subset.VALIDATION, Subset.TESTING}
        if self.ann_files[subset] is None:
            return False
        self.coco_dataset = CocoDataset(ann_file=self.ann_files[subset],
                                        data_root=self.data_roots[subset],
                                        classes=self.labels,
                                        test_mode=test_mode)
        return True

    def __getitem__(self, indx) -> dict:
        def create_gt_scored_label(label_name):
            return ScoredLabel(label=self.label_name_to_project_label(label_name))

        def create_gt_box(x1, y1, x2, y2, label):
            return Box(x1=x1, y1=y1, x2=x2, y2=y2, labels=[create_gt_scored_label(label)])

        item = self.coco_dataset[indx]
        divisor = np.tile([item['ori_shape'][:2][::-1]], 2)
        bboxes = item['gt_bboxes'] / divisor
        labels = item['gt_labels']

        shapes = [create_gt_box(*coords, self.labels[label_id]) for coords, label_id in zip(bboxes, labels)]

        image = Image(name=None, project=None, numpy=item['img'])
        annotation = Annotation(kind=AnnotationKind.ANNOTATION,
                                media_identifier=NullMediaIdentifier(),
                                shapes=shapes)
        datset_item = DatasetItem(image, annotation)
        return datset_item

    def __len__(self) -> int:
        assert self.coco_dataset is not None
        return len(self.coco_dataset)

    def get_labels(self) -> list:
        return self.labels

    def get_subset(self, subset: Subset) -> 'Dataset':
        dataset = deepcopy(self)
        if dataset.init_as_subset(subset):
            return dataset
        return NullDataset()
