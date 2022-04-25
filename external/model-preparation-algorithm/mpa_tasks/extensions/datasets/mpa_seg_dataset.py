# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmseg.datasets.builder import DATASETS
from segmentation_tasks.extension.datasets import OTEDataset
from mpa.utils.logger import get_logger

logger = get_logger()


@DATASETS.register_module()
class MPASegIncrDataset(OTEDataset):
    def __init__(self, **kwargs):
        self.img_indices = dict(old=[], new=[])
        pipeline = []
        if 'dataset' in kwargs:
            dataset = kwargs['dataset']
            if 'old_new_indices' in dataset:
                old_new_indices = dataset.old_new_indices
                self.img_indices['old'] = old_new_indices['old']
                self.img_indices['new'] = old_new_indices['new']
            ote_dataset = dataset.ote_dataset
            pipeline = dataset.pipeline
            classes = dataset.labels
        else:
            ote_dataset = kwargs['ote_dataset']
            pipeline = kwargs['pipeline']
            classes = kwargs['labels']

        for action in pipeline:
            if 'domain' in action:
                action.pop('domain')
        classes = [c.name for c in classes]
        classes = ['background'] + classes
        super().__init__(ote_dataset=ote_dataset, pipeline=pipeline, classes=classes)
        if self.label_map is None:
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in classes:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)
