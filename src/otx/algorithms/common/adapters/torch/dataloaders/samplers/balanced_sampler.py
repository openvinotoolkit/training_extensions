"""Balanced sampler for imbalanced data."""
import math

import numpy as np
from torch.utils.data.sampler import Sampler

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


class BalancedSampler(Sampler):  # pylint: disable=too-many-instance-attributes
    """Balanced sampler for imbalanced data for class-incremental task.

    This sampler is a sampler that creates an effective batch
    In reduce mode,
    reduce the iteration size by estimating the trials
    that all samples in the tail class are selected more than once with probability 0.999

    Args:
        dataset (Dataset): A built-up dataset
        samples_per_gpu (int): batch size of Sampling
        efficient_mode (bool): Flag about using efficient mode
    """

    def __init__(self, dataset, batch_size, efficient_mode=True, num_replicas=1, rank=0, drop_last=False):
        self.batch_size = batch_size
        self.repeat = 1
        if hasattr(dataset, "times"):
            self.repeat = dataset.times
        if hasattr(dataset, "dataset"):
            self.dataset = dataset.dataset
        else:
            self.dataset = dataset
        self.img_indices = self.dataset.img_indices
        self.num_cls = len(self.img_indices.keys())
        self.data_length = len(self.dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        if efficient_mode:
            # Reduce the # of sampling (sampling data for a single epoch)
            self.num_tail = min(len(cls_indices) for cls_indices in self.img_indices.values())
            base = 1 - (1 / self.num_tail)
            if base == 0:
                raise ValueError("Required more than one sample per class")
            self.num_trials = int(math.log(0.001, base))
            if int(self.data_length / self.num_cls) < self.num_trials:
                self.num_trials = int(self.data_length / self.num_cls)
        else:
            self.num_trials = int(self.data_length / self.num_cls)
        self.num_samples = self._calculate_num_samples()

        logger.info(f"This sampler will select balanced samples {self.num_trials} times")

    def _calculate_num_samples(self):
        num_samples = self.num_trials * self.num_cls * self.repeat

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
            for _ in range(self.num_trials):
                indice = np.concatenate(
                    [np.random.choice(self.img_indices[cls_indices], 1) for cls_indices in self.img_indices.keys()]
                )
                indices.append(indice)

        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()

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
            assert len(indices) == self.total_size

            # split and distribute indices
            len_indices = len(indices)
            indices = indices[
                self.rank * len_indices // self.num_replicas : (self.rank + 1) * len_indices // self.num_replicas
            ]

            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        """Return length of selected samples."""
        return self.num_samples
