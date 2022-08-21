# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmseg.datasets.builder import DATASETS
from segmentation_tasks.extension.datasets import OTEDataset
from mpa.utils.logger import get_logger
from mpa_tasks.utils.data_utils import get_old_new_img_indices

logger = get_logger()


@DATASETS.register_module()
class MPASegIncrDataset(OTEDataset):
    def __init__(self, **kwargs):
        pipeline = []
        test_mode = kwargs.get('test_mode', False)
        logger.info(f'test_mode : {test_mode}')
        if 'dataset' in kwargs:
            dataset = kwargs['dataset']
            ote_dataset = dataset.ote_dataset
            pipeline = dataset.pipeline
            classes = dataset.labels
            if test_mode is False:
                new_classes = dataset.new_classes
                self.img_indices = get_old_new_img_indices(classes, new_classes, ote_dataset)
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
