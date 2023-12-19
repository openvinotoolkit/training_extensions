# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LightningDataModule extension for OTX."""
from __future__ import annotations

import logging as log
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from datumaro import Dataset as DmDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from otx.core.config.data import DataModuleConfig, SubsetConfig
from otx.core.data.factory import OTXDatasetFactory
from otx.core.data.mem_cache import (
    MemCacheHandlerSingleton,
    parse_mem_cache_size_to_int,
)
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
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

        dataset = DmDataset.import_from(self.config.data_root, format=self.config.data_format)

        config_mapping = {
            self.config.train_subset.subset_name: self.config.train_subset,
            self.config.val_subset.subset_name: self.config.val_subset,
            self.config.test_subset.subset_name: self.config.test_subset,
        }

        for name, dm_subset in dataset.subsets().items():
            if name not in config_mapping:
                log.warning(f"{name} is not available. Skip it")
                continue

            self.subsets[name] = OTXDatasetFactory.create(
                task=self.task,
                dm_subset=dm_subset,
                cfg_subset=config_mapping[name],
                cfg_data_module=config,
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

    @classmethod
    def from_config(cls, config: dict | str | Path) -> OTXDataModule:
        """Create an instance of OTXDataModule from a configuration.

        Args:
            task (OTXTaskType): The task type.
            config (DataModuleConfig | str | Path): The configuration object or the path to the configuration file.

        Returns:
            OTXDataModule: An instance of OTXDataModule.

        """
        if isinstance(config, (str, Path)):
            with Path(config).open() as f:
                config = yaml.safe_load(f)["data"]
        if not isinstance(config, dict):
            msg = "Please double-check data config."
            raise TypeError(msg)
        task = config.pop("task")
        datamodule_config = config["config"]
        train_subset = datamodule_config.pop("train_subset", {})
        val_subset = datamodule_config.pop("val_subset", {})
        test_subset = datamodule_config.pop("test_subset", {})
        return cls(
            task=task,
            config=DataModuleConfig(
                train_subset=SubsetConfig(**train_subset),
                val_subset=SubsetConfig(**val_subset),
                test_subset=SubsetConfig(**test_subset),
                **datamodule_config,
            ),
        )

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
