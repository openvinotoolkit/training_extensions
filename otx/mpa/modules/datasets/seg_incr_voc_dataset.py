# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp

import numpy as np
from mmseg.datasets import DATASETS, CustomDataset

from otx.mpa.modules.utils.task_adapt import map_class_names


@DATASETS.register_module()
class SegIncrVOCDataset(CustomDataset):
    """Pascal VOC dataset for Class Incremental Learning.

    Args:
        split (str): Split txt file for Pascal VOC.
        classes (list): dataset classes
        new_classes (list): new classes
    """

    def __init__(self, split, classes, new_classes, **kwargs):
        super(SegIncrVOCDataset, self).__init__(
            img_suffix=".jpg", seg_map_suffix=".png", split=split, classes=classes, **kwargs
        )
        assert osp.exists(self.img_dir) and self.split is not None
        self.classes = classes
        self.new_classes = new_classes
        self.img_indices = dict(old=[], new=[])
        self.statistics()

    def statistics(self):
        gt_seg_maps = self.get_gt_seg_maps(False)
        classes = ["background"] + self.classes

        new_class_indices = map_class_names(self.new_classes, classes)
        for idx in range(len(gt_seg_maps)):
            gt_map = gt_seg_maps[idx]
            gt_map[np.where((gt_map == 255))] = 0
            gt = np.unique(gt_map)

            label_schema = []
            for i in gt:
                label_schema.append(classes[i])
            model2data = map_class_names(classes, label_schema)
            new_class_values = [model2data[idx] for idx in new_class_indices]
            if any(value is not -1 for value in new_class_values):
                self.img_indices["new"].append(idx)
            else:
                self.img_indices["old"].append(idx)
