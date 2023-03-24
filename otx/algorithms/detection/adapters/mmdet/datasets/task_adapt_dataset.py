"""Task Adapt Dataset for detection task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# import torch
import numpy as np
from mmdet.datasets import DATASETS, PIPELINES, build_dataset

from otx.algorithms.common.utils.task_adapt import (
    map_cat_and_cls_as_order,
    map_class_names,
)


# pylint: disable=invalid-name
@DATASETS.register_module()
class TaskAdaptEvalDataset:
    """Dataset wrapper for task-adaptive evaluation.

    Attributes:
        model_classes:
        **kwargs:

    Notes:
        TODO[Eugene/Jaeguk]: check if this class is still valid in OTX
    """

    def __init__(self, model_classes, **kwargs):
        dataset_cfg = kwargs.copy()
        org_type = dataset_cfg.pop("org_type")
        dataset_cfg["type"] = org_type
        self.dataset = build_dataset(dataset_cfg)
        self.model_classes = model_classes
        self.CLASSES = self.dataset.CLASSES
        self.data2model = map_class_names(self.CLASSES, self.model_classes)
        if org_type == "CocoDataset":
            self.dataset.cat2label, self.dataset.cat_ids = map_cat_and_cls_as_order(
                self.CLASSES, self.dataset.coco.cats
            )

    def __getitem__(self, idx):
        """Get item from TaskAdaptEvalDataset."""
        return self.dataset[idx]

    def __len__(self):
        """Length of TaskAdaptEvalDataset."""
        return len(self.dataset)

    def evaluate(self, results, **kwargs):
        """Filter & reorder detection results."""
        adapt_results = []
        for result in results:  # for each image
            adapt_result = []
            for model_class_index in self.data2model:  # for each class
                # Gather per-class results according to index mapping
                if model_class_index >= 0:
                    adapt_result.append(result[model_class_index])
                else:
                    adapt_result.append(np.empty([0, 5]))
            adapt_results.append(adapt_result)

        # Call evaluation w/ org arguments
        return self.dataset.evaluate(adapt_results, **kwargs)


@PIPELINES.register_module()
class AdaptClassLabels:
    """Data processor for task-adative annotation loading."""

    def __init__(self, src_classes, dst_classes):
        self.src2dst = map_class_names(src_classes, dst_classes)
        print("AdaptClassLabels")
        print("src_classes", src_classes)
        print("dst_classes", dst_classes)
        print("src2dst", self.src2dst)

    def __call__(self, data):
        """Call function of AdaptClassLabels."""
        src_labels = data["gt_labels"]
        dst_labels = []
        for src_label in src_labels:
            dst_labels.append(self.src2dst[src_label])
        data["gt_labels"] = np.array(dst_labels)
        return data
