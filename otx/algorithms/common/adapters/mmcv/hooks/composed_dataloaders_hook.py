# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Sequence, Union

from mmcv.runner import HOOKS, Hook
from torch.utils.data import DataLoader

from otx.mpa.modules.datasets.composed_dataloader import ComposedDL
from otx.mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class ComposedDataLoadersHook(Hook):
    def __init__(
        self,
        data_loaders: Union[Sequence[DataLoader], DataLoader],
    ):
        self.data_loaders = []
        self.composed_loader = None

        self.add_dataloaders(data_loaders)

    def add_dataloaders(self, data_loaders: Union[Sequence[DataLoader], DataLoader]):
        if isinstance(data_loaders, DataLoader):
            data_loaders = [data_loaders]
        else:
            data_loaders = list(data_loaders)

        self.data_loaders.extend(data_loaders)
        self.composed_loader = None

    def before_epoch(self, runner):
        if self.composed_loader is None:
            logger.info(
                "Creating ComposedDL "
                f"(runner's -> {runner.data_loader}, "
                f"hook's -> {[i for i in self.data_loaders]})"
            )
            self.composed_loader = ComposedDL([runner.data_loader, *self.data_loaders])
        # Per-epoch replacement: train-only loader -> train loader + additional loaders
        # (It's similar to local variable in epoch. Need to update every epoch...)
        runner.data_loader = self.composed_loader
