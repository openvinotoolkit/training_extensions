"""OTX sampler."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from typing import Sized

from torch.utils.data.sampler import Sampler
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


class OTXSampler(Sampler):  # pylint: disable=too-many-instance-attributes
    """Sampler that easily adapts to the dataset statistics. 

    In the exterme small dataset, the iteration per epoch could be set to 1 and then it could make slow training 
    since DataLoader reinitialized at every epoch. So, in the small dataset case, 
    OTXSampler repeats the dataset to enlarge the iterations per epoch.
    
    In the large dataset, the useful information is not totally linear relationship with the number of datasets.
    It closes to the log scale relationship, rather.
    
    So, this sampler samples or repeats the datasets acoording to the statistics of dataset.

    Args:
        dataset (Dataset): A built-up dataset
        samples_per_gpu (int): batch size of Sampling
        use_adaptive_repeats (bool): Flag about using adaptive repeats
    """

    def __init__(self, dataset: Sized, samples_per_gpu: int, use_adaptive_repeats: bool):
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        
        self.repeat = self._get_proper_repeats(use_adaptive_repeats)

    def _get_proper_repeats(self, use_adaptive_repeats: bool, coef=-0.7, min_repeat=1.0):
        """Calculate the proper repeats with considering the number of iterations."""
        n_repeats = 1
        if use_adaptive_repeats:
            # NOTE 
            # Currently, only support the integer type repeats. 
            # Will support the floating point repeats and large dataset cases.
            n_iters_per_epoch = math.ceil(len(self.dataset) / self.samples_per_gpu)
            n_repeats = math.floor(max(coef * math.sqrt(n_iters_per_epoch-1) + 5, min_repeat))
            logger.info("OTX Sampler: adaptive repeats enabled") 
            logger.info(f"OTX will use {n_repeats} times larger dataset made by repeated sampling")
        return n_repeats 