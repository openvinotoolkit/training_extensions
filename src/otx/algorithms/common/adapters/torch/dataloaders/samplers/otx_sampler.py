"""OTX sampler."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from otx.algorithms.common.adapters.mmcv.utils.config_utils import get_proper_repeat_times
from otx.algorithms.common.utils.task_adapt import unwrap_dataset
from otx.utils.logger import get_logger

logger = get_logger()


class OTXSampler(Sampler):  # pylint: disable=too-many-instance-attributes
    """Sampler that easily adapts to the dataset statistics.

    In the exterme small dataset, the iteration per epoch could be set to 1 and then it could make slow training
    since DataLoader reinitialized at every epoch. So, in the small dataset case,
    OTXSampler repeats the dataset to enlarge the iterations per epoch.

    In the large dataset, the useful information is not totally linear relationship with the number of datasets.
    It is close to the log scale relationship, rather.

    So, this sampler samples or repeats the datasets acoording to the statistics of dataset.

    Args:
        dataset (Dataset): A built-up dataset
        samples_per_gpu (int): batch size of Sampling
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): Flag about shuffling
        coef (int, optional): controls the repeat value
        min_repeat (float, optional): minimum value of the repeat dataset
        n_repeats (Union[float, int str], optional) : number of iterations for manual setting
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
    """

    def __init__(
        self,
        dataset: Dataset,
        samples_per_gpu: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        coef: float = -0.7,
        min_repeat: float = 1.0,
        n_repeats: Union[float, int, str] = "auto",
        seed: Optional[int] = None,
    ):

        self.dataset, _ = unwrap_dataset(dataset)
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        if n_repeats == "auto":
            repeat = get_proper_repeat_times(len(self.dataset), self.samples_per_gpu, coef, min_repeat)
        elif isinstance(n_repeats, (int, float)):
            repeat = float(n_repeats)
        else:
            raise ValueError(f"n_repeats: {n_repeats} should be auto or float or int value")
        # TODO: Currently, only supporting the int variable.
        # Will be removed.
        self.repeat = int(repeat)
        self.num_samples = math.ceil(len(self.dataset) * self.repeat / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        if seed is None:
            seed = np.random.randint(2**31)

        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        """Iter."""
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
        indices = [x for x in indices for _ in range(self.repeat)]
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample per rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # return up to num selected samples
        return iter(indices)

    def __len__(self):
        """Return length of selected samples."""
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
