# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LightningDataModule extension for OTX."""
from __future__ import annotations

import logging as log
from pathlib import Path
from typing import TYPE_CHECKING

from datumaro import Dataset as DmDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from otx.core.data.factory import OTXDatasetFactory
from otx.core.data.mem_cache import (
    MemCacheHandlerSingleton,
    parse_mem_cache_size_to_int,
)
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from otx.core.config.data import (
        DataModuleConfig,
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

        if self.config.data_format == "common_semantic_segmentation":
            # temporary hack, should be fixed later
            dataset = {}
            train_data_roots = Path(self.config.data_root) / "train"
            val_data_roots = Path(self.config.data_root) / "val"

            for subset_name, subset_path in zip(("train", "val", "test"),
                                           (train_data_roots, val_data_roots, val_data_roots)):
                dataset[subset_name] = DmDataset.import_from(subset_path,
                                                     format=self.config.data_format).subsets()["default"]
        else:
            dataset = DmDataset.import_from(self.config.data_root, format=self.config.data_format).subsets()

        config_mapping = {
            self.config.train_subset.subset_name: self.config.train_subset,
            self.config.val_subset.subset_name: self.config.val_subset,
            self.config.test_subset.subset_name: self.config.test_subset,
        }

        for name, dm_subset in dataset.items():
            if name not in config_mapping:
                log.warning(f"{name} is not available. Skip it")
                continue

            self.subsets[name] = OTXDatasetFactory.create(
                task=self.task,
                dm_subset=dm_subset,
                config=config_mapping[name],
            )
            log.info(f"Add name: {name}, self.subsets: {self.subsets}")

        mem_size = parse_mem_cache_size_to_int(config.mem_cache_size)
        mem_cache_mode = (
            "singleprocessing"
            if all(config.num_workers == 0 for config in config_mapping.values())
            else "multiprocessing"
        )

        self.mem_cache_handler = MemCacheHandlerSingleton.create(
            mode=mem_cache_mode,
            mem_size=mem_size,
        )

    def __del__(self) -> None:
        MemCacheHandlerSingleton.delete()

    def _get_dataset(self, subset: str) -> OTXDataset:
        if (dataset := self.subsets.get(subset)) is None:
            msg = f"Dataset has no '{subset}'. Available subsets = {list(self.subsets.keys())}"
            raise KeyError(msg)
        return dataset

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        config = self.config.train_subset
        dataset = self._get_dataset(config.subset_name)

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
            persistent_workers=config.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Get val dataloader."""
        config = self.config.val_subset
        dataset = self._get_dataset(config.subset_name)

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
            persistent_workers=config.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        config = self.config.test_subset
        dataset = self._get_dataset(config.subset_name)

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
            persistent_workers=config.num_workers > 0,
        )

    def setup(self, stage: str) -> None:
        """Setup for each stage."""

    def teardown(self, stage: str) -> None:
        """Teardown for each stage."""
        # clean up after fit or test
        # called on every process in DDP
