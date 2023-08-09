"""Adaptive repeat data hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook, get_dist_info
from torch.utils.data import DataLoader

from otx.algorithms.common.adapters.torch.dataloaders.samplers import OTXSampler
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class AdaptiveRepeatDataHook(Hook):
    """Hook that adaptively repeats the dataset to control the number of iterations.

    Args:
        coef (float) : coefficient that effects to number of repeats
                       (coef * math.sqrt(num_iters-1)) +5
        min_repeat (float) : minimum repeats
    """

    def __init__(self, coef: float = -0.7, min_repeat: float = 1.0):
        self.coef = coef
        self.min_repeat = min_repeat

    def before_epoch(self, runner):
        """Convert to OTX Sampler."""
        dataset = runner.data_loader.dataset
        batch_size = runner.data_loader.batch_size
        num_workers = runner.data_loader.num_workers
        collate_fn = runner.data_loader.collate_fn
        worker_init_fn = runner.data_loader.worker_init_fn
        rank, world_size = get_dist_info()

        sampler = OTXSampler(
            dataset=dataset,
            samples_per_gpu=batch_size,
            use_adaptive_repeats=True,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            coef=self.coef,
            min_repeat=self.min_repeat,
        )

        runner.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
        )
