"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function

import os
import torch
from torch import distributed as dist
from torch.utils.data import Sampler

from examples.common.example_logger import logger


def configure_distributed(config):
    if config.dist_url == "env://" and config.rank == -1:
        config.rank = int(os.environ["RANK"])
    config.ngpus_per_node = torch.cuda.device_count()

    if config.current_gpu is not None:
        # Distributed multiprocessing
        config.rank = config.rank * config.ngpus_per_node + config.current_gpu

    logger.info('| distributed init (rank {}): {}'.format(
        config.rank, config.dist_url))
    dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                            world_size=config.world_size, rank=config.rank)
    config.world_size = dist.get_world_size()


class DistributedSampler(Sampler):
    def __init__(self, dataset, rank=None, world_size=None):
        super().__init__(dataset)
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.world_size = world_size
        self.rank = rank
        indices = list(range(len(dataset)))
        self.samples_per_rank = (len(indices) - 1) // self.world_size + 1
        self.indices = indices[self.rank * self.samples_per_rank: (self.rank + 1) * self.samples_per_rank]

        if len(self.indices) < self.samples_per_rank:
            # Workaround for mock datasets with a small number of entries
            pad = [0] * (self.samples_per_rank - len(self.indices))
            self.indices += pad

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
