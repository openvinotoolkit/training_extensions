# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class incremental sampler for cls-incremental learning."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Sampler

from otx.core.data.dataset.base import OTXDataset
from otx.core.utils.utils import get_idx_list_per_classes


class ClassIncrementalSampler(Sampler):
    """Sampler for Class-Incremental Task.

    This sampler is a sampler that creates an effective batch
    For default setting,
    the square root of (number of old data/number of new data) is used as the ratio of old data
    In effective mode,
    the ratio of old and new data is used as 1:1

    Args:
        dataset (OTXDataset): A built-up dataset
        batch_size (int): batch size of Sampling
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
        dataset: OTXDataset,
        batch_size: int,
        old_classes: list[str],
        new_classes: list[str],
        efficient_mode: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        n_repeats: int = 1,
        generator: torch.Generator | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.generator = generator
        self.repeat = n_repeats

        super().__init__(dataset)

        # Need to split new classes dataset indices & old classses dataset indices
        ann_stats = get_idx_list_per_classes(dataset.dm_subset, True)
        new_indices, old_indices = [], []
        for cls in new_classes:
            new_indices.extend(ann_stats[cls])
        self.new_indices = torch.tensor(new_indices, dtype=torch.int64)
        for cls in old_classes:
            old_indices.extend(ann_stats[cls])
        self.old_indices = torch.tensor(old_indices, dtype=torch.int64)

        if not len(self.new_indices) > 0:
            self.new_indices = self.old_indices
            self.old_indices = torch.tensor([], dtype=torch.int64)

        old_new_ratio = np.sqrt(len(self.old_indices) / len(self.new_indices))

        if efficient_mode:
            self.data_length = int(len(self.new_indices) * (1 + old_new_ratio))
            self.old_new_ratio = 1
        else:
            self.data_length = len(self.dataset)
            self.old_new_ratio = int(old_new_ratio)

        self.num_samples = self._calcuate_num_samples()

    def _calcuate_num_samples(self) -> int:
        num_samples = self.repeat * (1 + self.old_new_ratio) * int(self.data_length / (1 + self.old_new_ratio))

        if not self.drop_last:
            num_samples += (
                int(np.ceil(self.data_length * self.repeat / self.batch_size)) * self.batch_size - num_samples
            )

        if self.num_replicas > 1:
            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and num_samples % self.num_replicas != 0:
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (num_samples - self.num_replicas) / self.num_replicas,
                )
            else:
                num_samples = math.ceil(num_samples / self.num_replicas)
            self.total_size = num_samples * self.num_replicas

        return num_samples

    def __iter__(self):
        """Iter."""
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        indices = []
        for _ in range(self.repeat):
            num_batches = self.data_length // self.batch_size
            for _ in range(num_batches):
                num_new_per_batch = self.batch_size // (1 + self.old_new_ratio)

                new_indices_random = torch.randint(0, len(self.new_indices), (num_new_per_batch,), generator=generator)
                old_indices_random = torch.randint(
                    0,
                    len(self.old_indices),
                    (self.batch_size - num_new_per_batch,),
                    generator=generator,
                )

                new_samples = self.new_indices[new_indices_random]
                old_samples = self.old_indices[old_indices_random]

                indices.append(torch.cat([new_samples, old_samples]))

        indices = torch.cat(indices)
        if not self.drop_last:
            num_extra = int(
                np.ceil(self.data_length * self.repeat / self.batch_size),
            ) * self.batch_size - len(indices)
            indices = torch.cat(
                [
                    indices,
                    indices[torch.randint(0, len(indices), (num_extra,), generator=generator)],
                ],
            )

        if self.num_replicas > 1:
            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                # add extra samples to make it evenly divisible
                if padding_size <= len(indices):
                    indices = torch.cat([indices, indices[:padding_size]])
                else:
                    indices = torch.cat([indices, (indices * math.ceil(padding_size / len(indices)))[:padding_size]])
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[: self.total_size]

            # shuffle before distributing indices
            indices = indices[torch.randperm(len(indices))]

            # subsample
            indices = indices[self.rank : self.total_size : self.num_replicas]

        return iter(indices)

    def __len__(self):
        """Return length of selected samples."""
        return self.num_samples
