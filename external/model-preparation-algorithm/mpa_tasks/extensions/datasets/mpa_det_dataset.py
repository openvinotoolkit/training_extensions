# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.datasets.builder import DATASETS
from detection_tasks.extension.datasets import OTEDataset
from mpa.utils.logger import get_logger

logger = get_logger()


@DATASETS.register_module()
class MPADetDataset(OTEDataset):
    def __init__(self, old_new_indices=None, **kwargs):
        if old_new_indices is not None:
            self.old_new_indices = old_new_indices
            self.img_indices = dict(old=self.old_new_indices['old'], new=self.old_new_indices['new'])
        dataset_cfg = kwargs.copy()
        _ = dataset_cfg.pop('org_type', None)
        super().__init__(**dataset_cfg)

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.get_ann_info(idx)['labels'].astype(int).tolist()
