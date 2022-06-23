# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.datasets.builder import DATASETS
from detection_tasks.extension.datasets import OTEDataset
from mpa.utils.logger import get_logger
from mpa_tasks.utils.data_utils import get_old_new_img_indices

logger = get_logger()


@DATASETS.register_module()
class MPADetDataset(OTEDataset):
    def __init__(self, **kwargs):
        dataset_cfg = kwargs.copy()
        _ = dataset_cfg.pop('org_type', None)
        if dataset_cfg.get('new_classes', False):
            new_classes = dataset_cfg.pop('new_classes')
        super().__init__(**dataset_cfg)

        test_mode = kwargs.get('test_mode', False)
        if test_mode is False:
            self.img_indices = get_old_new_img_indices(self.labels, new_classes, self.ote_dataset)

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.get_ann_info(idx)['labels'].astype(int).tolist()
