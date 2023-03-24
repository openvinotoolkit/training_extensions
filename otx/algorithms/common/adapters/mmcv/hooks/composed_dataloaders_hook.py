"""Composed dataloader hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Sequence, Union

from mmcv.runner import HOOKS, Hook
from torch.utils.data import DataLoader

from otx.algorithms.common.adapters.torch.dataloaders import ComposedDL
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class ComposedDataLoadersHook(Hook):
    """Composed dataloader hook, which makes a composed dataloader which can combine multiple data loaders.

    Especially used for semi-supervised learning to aggregate a unlabeled dataloader and a labeled dataloader.
    """

    def __init__(
        self,
        data_loaders: Union[Sequence[DataLoader], DataLoader],
    ):
        self.data_loaders = []  # type: List[DataLoader]
        self.composed_loader = None

        self.add_dataloaders(data_loaders)

    def add_dataloaders(self, data_loaders: Union[Sequence[DataLoader], DataLoader]):
        """Create data_loaders to be added into composed dataloader."""
        if isinstance(data_loaders, DataLoader):
            data_loaders = [data_loaders]
        else:
            data_loaders = list(data_loaders)

        self.data_loaders.extend(data_loaders)
        self.composed_loader = None

    def before_epoch(self, runner):
        """Create composedDL before running epoch."""
        if self.composed_loader is None:
            logger.info("Creating ComposedDL " f"(runner's -> {runner.data_loader}, " f"hook's -> {self.data_loaders})")
            self.composed_loader = ComposedDL([runner.data_loader, *self.data_loaders])
        # Per-epoch replacement: train-only loader -> train loader + additional loaders
        # (It's similar to local variable in epoch. Need to update every epoch...)
        runner.data_loader = self.composed_loader
