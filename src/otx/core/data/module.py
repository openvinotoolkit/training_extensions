# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LightningDataModule extension for OTX."""
from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING

from datumaro import Dataset as DmDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from otx.core.types.task import OTXTaskType

from .factory import OTXDatasetFactory

if TYPE_CHECKING:
    from otx.core.config.data import (
        DataModuleConfig,
        SubsetConfig,
    )

    from .dataset.base import OTXDataset


class OTXDataModule(LightningDataModule):
    """LightningDataModule extension for OTX pipeline."""

    def __init__(self, task: OTXTaskType, config: DataModuleConfig) -> None:
        """Constructor."""
        super().__init__()
        self.task = task
        self.config = config
        self.subsets: dict[str, OTXDataset] = {}
        self.save_hyperparameters()

        dataset = DmDataset.import_from(
            self.config.data_root,
            format=self.config.data_format,
        )

        available_name_mapping = {
            self.config.train_subset_name: "train",
            self.config.val_subset_name: "val",
            self.config.test_subset_name: "test",
        }

        for name, dm_subset in dataset.subsets().items():
            if name not in available_name_mapping:
                log.warning(f"{name} is not available. Skip it")
                continue

            sub_config = self._get_config(available_name_mapping[name])

            self.subsets[name] = OTXDatasetFactory.create(
                task=self.task,
                dm_subset=dm_subset,
                config=sub_config,
            )
            log.info(f"Add name: {name}, self.subsets: {self.subsets}")

    def _get_config(self, subset: str) -> SubsetConfig:
        if (config := self.config.subsets.get(subset)) is None:
            msg = f"Config has no '{subset}' subset configuration"
            raise KeyError(msg)

        return config

    def _get_dataset(self, subset: str) -> OTXDataset:
        if (dataset := self.subsets.get(subset)) is None:
            msg = (
                f"Dataset has no '{subset}'. Available subsets = {self.subsets.keys()}"
            )
            raise KeyError(msg)
        return dataset

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        config = self._get_config("train")
        dataset = self._get_dataset(self.config.train_subset_name)

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Get val dataloader."""
        config = self._get_config("val")
        dataset = self._get_dataset(self.config.val_subset_name)

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        config = self._get_config("test")
        dataset = self._get_dataset(self.config.test_subset_name)

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    def setup(self, stage: str) -> None:
        """Setup for each stage."""

    def teardown(self, stage: str) -> None:
        """Teardown for each stage."""
        # clean up after fit or test
        # called on every process in DDP
