"""Adaptive repeat data hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmengine.dist import get_dist_info
from torch.utils.data import DataLoader

from otx.v2.adapters.torch.modules.utils.repeat_times import get_proper_repeat_times
from otx.v2.adapters.torch.modules.dataloaders.samplers import OTXSampler
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class AdaptiveRepeatDataHook(Hook):
    """Hook that adaptively repeats the dataset to control the number of iterations.

    Args:
        train_batch_size (int) : The batch size of the train dataloader
        train_data_size (int) : The number of the training dataset
        coef (float, optional) : coefficient that effects to number of repeats
                       (coef * math.sqrt(num_iters-1)) +5
        min_repeat (float, optional) : minimum repeats
    """

    def __init__(self, coef: float = -0.7, min_repeat: float = 1.0) -> None:
        self.coef = coef
        self.min_repeat = min_repeat

        self._initialized = False

    def initialize(self) -> None:
        self.n_repeats = get_proper_repeat_times(self.data_size, self.batch_size, self.coef, self.min_repeat)
        self.rank, self.world_size = get_dist_info()

    def before_run(self, runner):
        """Change the runner's max_iter."""
        if not self._initialized and hasattr(runner, "train_dataloader"):
            self.data_size = len(runner.train_dataloader.dataset)
            self.batch_size = runner.train_dataloader.batch_size
            self.initialize()
            self._initialized = True

        if self.n_repeats > 1:
            iter_per_epoch = int(self.data_size / self.batch_size)

            logger.info("Adaptive repeat is enabled")
            logger.info(f"- Repeat times: {self.n_repeats}")
            logger.info(f"- Batch size: {self.batch_size}")
            logger.info(f"- Num iters per epoch: {iter_per_epoch} -> {iter_per_epoch * self.n_repeats}")
            logger.info(f"- Total iters: {runner.max_iters} -> {runner.max_iters * self.n_repeats}")

            runner._max_iters = int(runner.max_iters * self.n_repeats)

    def before_epoch(self, runner: Runner) -> None:
        """Convert to OTX Sampler."""
        dataset = runner.data_loader.dataset
        num_workers = runner.data_loader.num_workers
        collate_fn = runner.data_loader.collate_fn
        worker_init_fn = runner.data_loader.worker_init_fn

        sampler = OTXSampler(
            dataset=dataset,
            samples_per_gpu=self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            coef=self.coef,
            min_repeat=self.min_repeat,
            n_repeats=self.n_repeats,
        )

        runner.data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
        )
