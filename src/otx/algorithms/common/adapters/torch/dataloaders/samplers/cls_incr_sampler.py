"""Class incremental sampler for cls-incremental learning."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
import random
from typing import Union

import numpy as np
from torch.utils.data import Dataset

from .otx_sampler import OTXSampler


class ClsIncrSampler(OTXSampler):  # pylint: disable=too-many-instance-attributes
    """Sampler for Class-Incremental Task.

    This sampler is a sampler that creates an effective batch
    For default setting,
    the square root of (number of old data/number of new data) is used as the ratio of old data
    In effective mode,
    the ratio of old and new data is used as 1:1

    Args:
        dataset (Dataset): A built-up dataset
        samples_per_gpu (int): batch size of Sampling
        efficient_mode (bool): Flag about using efficient mode
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        n_repeats (Union[float, int, str], optional) : number of iterations for manual setting
    """

    def __init__(
        self,
        dataset: Dataset,
        samples_per_gpu: int,
        efficient_mode: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        n_repeats: Union[float, int, str] = 1,
    ):
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        super().__init__(dataset, samples_per_gpu, n_repeats=n_repeats)

        if hasattr(self.dataset, "img_indices"):
            self.new_indices = self.dataset.img_indices["new"]
            self.old_indices = self.dataset.img_indices["old"]
        else:
            raise TypeError(f"{self.dataset} type does not have img_indices")

        if not len(self.new_indices) > 0:
            self.new_indices = self.old_indices
            self.old_indices = []

        old_new_ratio = np.sqrt(len(self.old_indices) / len(self.new_indices))

        if efficient_mode:
            self.data_length = int(len(self.new_indices) * (1 + old_new_ratio))
            self.old_new_ratio = 1
        else:
            self.data_length = len(self.dataset)
            self.old_new_ratio = int(old_new_ratio)

        self.num_samples = self._calcuate_num_samples()

    def _calcuate_num_samples(self):
        num_samples = self.repeat * (1 + self.old_new_ratio) * int(self.data_length / (1 + self.old_new_ratio))

        if not self.drop_last:
            num_samples += (
                int(np.ceil(self.data_length * self.repeat / self.samples_per_gpu)) * self.samples_per_gpu - num_samples
            )

        if self.num_replicas > 1:
            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and num_samples % self.num_replicas != 0:  # type: ignore
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (num_samples - self.num_replicas)
                    / self.num_replicas  # type: ignore
                )
            else:
                num_samples = math.ceil(num_samples / self.num_replicas)  # type: ignore
            self.total_size = num_samples * self.num_replicas

        return num_samples

    def __iter__(self):
        """Iter."""
        indices = []
        for _ in range(self.repeat):
            for _ in range(int(self.data_length / (1 + self.old_new_ratio))):
                indice = np.concatenate(
                    [np.random.choice(self.new_indices, 1), np.random.choice(self.old_indices, self.old_new_ratio)]
                )
                indices.append(indice)

        indices = np.concatenate(indices)
        if not self.drop_last:
            num_extra = int(
                np.ceil(self.data_length * self.repeat / self.samples_per_gpu)
            ) * self.samples_per_gpu - len(indices)
            indices = np.concatenate([indices, np.random.choice(indices, num_extra)])
        indices = indices.astype(np.int64).tolist()

        if self.num_replicas > 1:
            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                # add extra samples to make it evenly divisible
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[: self.total_size]
            assert len(indices) == self.total_size

            # shuffle before distributing indices
            random.shuffle(indices)

            # subsample
            indices = indices[self.rank : self.total_size : self.num_replicas]
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        """Return length of selected samples."""
        return self.num_samples
