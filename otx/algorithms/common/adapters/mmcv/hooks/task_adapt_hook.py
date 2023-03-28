"""Task adapt hook which selects a proper sampler."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook, get_dist_info
from torch.utils.data import DataLoader

from otx.algorithms.common.adapters.torch.dataloaders.samplers import (
    BalancedSampler,
    ClsIncrSampler,
)
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class TaskAdaptHook(Hook):
    """Task Adaptation Hook for Task-Inc & Class-Inc.

    Args:
        src_classes (list): A list of old classes used in the existing model
        dst_classes (list): A list of classes including new_classes to be newly learned
        model_type (str): Types of models used for learning
        sampler_flag (bool): Flag about using ClsIncrSampler
        efficient_mode (bool): Flag about using efficient mode sampler
    """

    def __init__(
        self,
        src_classes,
        dst_classes,
        model_type="FasterRCNN",
        sampler_flag=False,
        sampler_type="cls_incr",
        efficient_mode=False,
    ):
        self.src_classes = src_classes
        self.dst_classes = dst_classes
        self.model_type = model_type
        self.sampler_flag = sampler_flag
        self.sampler_type = sampler_type
        self.efficient_mode = efficient_mode

        logger.info(f"Task Adaptation: {self.src_classes} => {self.dst_classes}")
        logger.info(f"- Efficient Mode: {self.efficient_mode}")
        logger.info(f"- Sampler type: {self.sampler_type}")
        logger.info(f"- Sampler flag: {self.sampler_flag}")

    def before_epoch(self, runner):
        """Produce a proper sampler for task-adaptation."""
        if self.sampler_flag:
            dataset = runner.data_loader.dataset
            batch_size = runner.data_loader.batch_size
            num_workers = runner.data_loader.num_workers
            collate_fn = runner.data_loader.collate_fn
            worker_init_fn = runner.data_loader.worker_init_fn
            rank, world_size = get_dist_info()
            if self.sampler_type == "balanced":
                sampler = BalancedSampler(
                    dataset, batch_size, efficient_mode=self.efficient_mode, num_replicas=world_size, rank=rank
                )
            else:
                sampler = ClsIncrSampler(
                    dataset, batch_size, efficient_mode=self.efficient_mode, num_replicas=world_size, rank=rank
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
