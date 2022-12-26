# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.datasets import DATASETS, build_dataset, ClassBalancedDataset
import numpy as np
import math

from otx.mpa.utils.logger import get_logger

logger = get_logger()


@DATASETS.register_module()
class PseudoBalancedDataset(ClassBalancedDataset):
    """Pseudo-length Wrapper for ClassBalancedDataset
    """
    def __init__(
        self,
        pseudo_length=0,
        oversample_thr=0.001,
        filter_empty_gt=True,
        **kwargs
    ):
        # Original dataset
        dataset_cfg = kwargs.copy()
        org_type = dataset_cfg.pop('org_type')
        dataset_cfg['type'] = org_type
        dataset = build_dataset(dataset_cfg)

        # ClassBalancedDataset
        super().__init__(
            dataset=dataset,
            oversample_thr=oversample_thr,
            filter_empty_gt=filter_empty_gt
        )

        # Pseudo length
        self.org_length = len(dataset)
        self.balanced_length = len(self.repeat_indices)
        if pseudo_length == 0.0:
            # Original data size
            self.pseudo_length = self.org_length
        elif pseudo_length < 0.0:
            # Regarding information magnitude is proportional to sqrt(len(dataset))
            self.pseudo_length = int(10*math.sqrt(float(self.balanced_length)))
        else:
            # Manual length setting
            self.pseudo_length = pseudo_length
        logger.info('PseudoBalancedDataset ----------------------')
        logger.info(f'--> org_length: {self.org_length}')
        logger.info(f'--> balanced_length: {self.balanced_length}')
        logger.info(f'--> pseudo_length: {self.pseudo_length}')

        # Aspect ratio group
        # TODO: not sure yet how to deal w/ static aspect ratio flag
        # There might be some inefficiency due to mixed aspect ratio
        self.flag = np.zeros((self.pseudo_length,))

    def __getitem__(self, idx):
        pseudo_idx = (idx + np.random.randint(self.balanced_length)) % self.balanced_length
        return super().__getitem__(pseudo_idx)

    def __len__(self):
        return self.pseudo_length
