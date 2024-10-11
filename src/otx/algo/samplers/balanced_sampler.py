# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Balanced sampler for imbalanced data."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Sampler

from otx.core.utils.utils import get_idx_list_per_classes

if TYPE_CHECKING:
    from otx.core.data.dataset.base import OTXDataset


class BalancedSampler(Sampler):
    """Balanced sampler for imbalanced data for class-incremental task.

    This sampler is a sampler that creates an effective batch
    In reduce mode,
    reduce the iteration size by estimating the trials
    that all samples in the tail class are selected more than once with probability 0.999

    Args:
        dataset (OTXDataset): A built-up dataset
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
        n_repeats (int, optional) : number of iterations for manual setting
    """

    def __init__(
        self,
        dataset: OTXDataset,
        efficient_mode: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        n_repeats: int = 1,
        generator: torch.Generator | None = None,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.generator = generator
        self.repeat = n_repeats

        super().__init__(dataset)

        # img_indices: dict[label: list[idx]]
        ann_stats = get_idx_list_per_classes(dataset.dm_subset)
        self.img_indices = {k: torch.tensor(v, dtype=torch.int64) for k, v in ann_stats.items() if len(v) > 0}
        self.num_cls = len(self.img_indices.keys())
        self.data_length = len(self.dataset)
        self.num_trials = max(int(self.data_length / self.num_cls), 1)

        if efficient_mode:
            # Reduce the # of sampling (sampling data for a single epoch)
            num_tail = min(len(cls_indices) for cls_indices in self.img_indices.values())
            if num_tail > 1:
                base = 1 - (1 / num_tail)
                num_reduced_trials = int(math.log(0.001, base))
                self.num_trials = min(num_reduced_trials, self.num_trials)

        self.num_samples = self._calculate_num_samples()

    def _calculate_num_samples(self) -> int:
        num_samples = self.num_trials * self.num_cls * self.repeat

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
            for _ in range(self.num_trials):
                index = torch.cat(
                    [
                        self.img_indices[cls_indices][
                            torch.randint(0, len(self.img_indices[cls_indices]), (1,), generator=self.generator)
                        ]
                        for cls_indices in self.img_indices
                    ],
                )
                indices.append(index)

        indices = torch.cat(indices)
        indices = indices.tolist()

        if self.num_replicas > 1:
            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[: self.total_size]

            # split and distribute indices
            len_indices = len(indices)
            indices = indices[
                self.rank * len_indices // self.num_replicas : (self.rank + 1) * len_indices // self.num_replicas
            ]

        return iter(indices)

    def __len__(self):
        """Return length of selected samples."""
        return self.num_samples
