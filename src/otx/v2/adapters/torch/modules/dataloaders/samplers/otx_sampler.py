"""OTX sampler."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from otx.v2.adapters.torch.modules.utils.task_adapt import unwrap_dataset
from otx.v2.api.utils.logger import get_logger

if TYPE_CHECKING:
    from torch.utils.data import Dataset

logger = get_logger()


class OTXSampler(Sampler):
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
        use_adaptive_repeats (bool): Flag about using adaptive repeats
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): Flag about shuffling
        coef (int, optional): controls the repeat value
        min_repeat (float, optional): minimum value of the repeat dataset
    """

    def __init__(
        self,
        dataset: Dataset,
        samples_per_gpu: int,
        use_adaptive_repeats: bool,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        coef: float = -0.7,
        min_repeat: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initializes an OTXSampler object.

        Args:
            dataset (Dataset): The dataset to sample from.
            samples_per_gpu (int): The number of samples per GPU.
            use_adaptive_repeats (bool): Whether to use adaptive repeats.
            num_replicas (int, optional): The number of replicas. Defaults to 1.
            rank (int, optional): The rank. Defaults to 0.
            shuffle (bool, optional): Whether to shuffle the samples. Defaults to True.
            coef (float, optional): The coefficient. Defaults to -0.7.
            min_repeat (float, optional): The minimum repeat. Defaults to 1.0.
            seed (int, optional): The random seed. Defaults to None.
        """
        self.dataset, _ = unwrap_dataset(dataset)
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.repeat = self._get_proper_repeats(use_adaptive_repeats, coef, min_repeat)

        self.num_samples = math.ceil(len(self.dataset) * self.repeat / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        if seed is None:
            seed = int(np.random.default_rng().integers(2**31))

        self.seed = seed
        self.epoch = 0

    def _get_proper_repeats(self, use_adaptive_repeats: bool, coef: float, min_repeat: float) -> int:
        """Calculate the proper repeats with considering the number of iterations."""
        n_repeats = 1
        if use_adaptive_repeats:
            # NOTE
            # Currently, only support the integer type repeats.
            # Will support the floating point repeats and large dataset cases.
            n_iters_per_epoch = math.ceil(len(self.dataset) / self.samples_per_gpu)
            n_repeats = math.floor(max(coef * math.sqrt(n_iters_per_epoch - 1) + 5, min_repeat))
            logger.info("OTX Sampler: adaptive repeats enabled")
            logger.info(f"OTX will use {n_repeats} times larger dataset made by repeated sampling")
        return n_repeats

    def __iter__(self) -> Iterator:
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

        # subsample per rank
        indices = indices[self.rank : self.total_size : self.num_replicas]

        # return up to num selected samples
        return iter(indices)

    def __len__(self) -> int:
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
