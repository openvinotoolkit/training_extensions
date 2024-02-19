# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LightningDataModule extension for OTX."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING

from datumaro import Dataset as DmDataset
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler

from otx.core.data.dataset.base import LabelInfo
from otx.core.data.dataset.tile import OTXTileDatasetFactory
from otx.core.data.factory import OTXDatasetFactory
from otx.core.data.mem_cache import (
    MemCacheHandlerSingleton,
    parse_mem_cache_size_to_int,
)
from otx.core.data.pre_filtering import pre_filtering
from otx.core.data.tile_adaptor import adapt_tile_config
from otx.core.types.device import DeviceType
from otx.core.types.task import OTXTaskType
from otx.core.utils.utils import get_adaptive_num_workers

if TYPE_CHECKING:
    from lightning.pytorch.utilities.parsing import AttributeDict

    from otx.core.config.data import DataModuleConfig
    from otx.core.data.dataset.base import OTXDataset


class OTXDataModule(LightningDataModule):
    """LightningDataModule extension for OTX pipeline."""

    def __init__(
        self,
        task: OTXTaskType,
        config: DataModuleConfig,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.task = task
        self.config = config
        self.subsets: dict[str, OTXDataset] = {}
        self.save_hyperparameters()

        # TODO (Jaeguk): This is workaround for a bug in Datumaro.
        # These lines should be removed after next datumaro release.
        # https://github.com/openvinotoolkit/datumaro/pull/1223/files
        from datumaro.plugins.data_formats.video import VIDEO_EXTENSIONS

        VIDEO_EXTENSIONS.append(".mp4")

        dataset = DmDataset.import_from(self.config.data_root, format=self.config.data_format)
        if self.task != "H_LABEL_CLS":
            dataset = pre_filtering(dataset, self.config.data_format)
        if config.tile_config.enable_tiler and config.tile_config.enable_adaptive_tiling:
            adapt_tile_config(config.tile_config, dataset=dataset)

        config_mapping = {
            self.config.train_subset.subset_name: self.config.train_subset,
            self.config.val_subset.subset_name: self.config.val_subset,
            self.config.test_subset.subset_name: self.config.test_subset,
        }

        if self.config.auto_num_workers:
            if self.config.device not in [DeviceType.gpu, DeviceType.auto]:
                log.warning(
                    "Only GPU device type support auto_num_workers. "
                    f"Current deveice type is {self.config.device!s}. auto_num_workers is skipped.",
                )
            elif (num_workers := get_adaptive_num_workers()) is not None:
                for subset_name, subset_config in config_mapping.items():
                    log.info(
                        f"num_workers of {subset_name} subset is changed : "
                        f"{subset_config.num_workers} -> {num_workers}",
                    )
                    subset_config.num_workers = num_workers

        mem_size = parse_mem_cache_size_to_int(config.mem_cache_size)
        mem_cache_mode = (
            "singleprocessing"
            if all(config.num_workers == 0 for config in config_mapping.values())
            else "multiprocessing"
        )
        mem_cache_handler = MemCacheHandlerSingleton.create(
            mode=mem_cache_mode,
            mem_size=mem_size,
        )

        meta_infos: list[LabelInfo] = []
        for name, dm_subset in dataset.subsets().items():
            if name not in config_mapping:
                log.warning(f"{name} is not available. Skip it")
                continue

            dataset = OTXDatasetFactory.create(
                task=self.task,
                dm_subset=dm_subset,
                mem_cache_handler=mem_cache_handler,
                cfg_subset=config_mapping[name],
                cfg_data_module=config,
            )

            if config.tile_config.enable_tiler:
                dataset = OTXTileDatasetFactory.create(
                    task=self.task,
                    dataset=dataset,
                    tile_config=config.tile_config,
                )
            self.subsets[name] = dataset

            meta_infos += [self.subsets[name].meta_info]
            log.info(f"Add name: {name}, self.subsets: {self.subsets}")

        if self._is_meta_info_valid(meta_infos) is False:
            msg = "All data meta infos of subsets should be the same."
            raise ValueError(msg)

        self.meta_info = next(iter(meta_infos))

    def _is_meta_info_valid(self, meta_infos: list[LabelInfo]) -> bool:
        """Check whether there are mismatches in the metainfo for the all subsets."""
        if all(meta_info == meta_infos[0] for meta_info in meta_infos):
            return True
        return False

    def _get_dataset(self, subset: str) -> OTXDataset:
        if (dataset := self.subsets.get(subset)) is None:
            msg = f"Dataset has no '{subset}'. Available subsets = {list(self.subsets.keys())}"
            raise KeyError(msg)
        return dataset

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        config = self.config.train_subset
        dataset = self._get_dataset(config.subset_name)

        common_args = {
            "dataset": dataset,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": True,
            "collate_fn": dataset.collate_fn,
            "persistent_workers": config.num_workers > 0,
        }

        tile_config = self.config.tile_config
        if tile_config.enable_tiler and tile_config.sampling_ratio < 1:
            num_samples = max(1, int(len(dataset) * tile_config.sampling_ratio))
            log.info(f"Using tiled sampling with {num_samples} samples")
            common_args.update(
                {
                    "shuffle": False,
                    "sampler": RandomSampler(dataset, num_samples=num_samples),
                },
            )
        else:
            common_args["shuffle"] = True
        return DataLoader(**common_args)

    def val_dataloader(self) -> DataLoader:
        """Get val dataloader."""
        config = self.config.val_subset
        dataset = self._get_dataset(config.subset_name)

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
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
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            persistent_workers=config.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        config = self.config.test_subset
        dataset = self._get_dataset(config.subset_name)

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            persistent_workers=config.num_workers > 0,
        )

    def setup(self, stage: str) -> None:
        """Setup for each stage."""

    def teardown(self, stage: str) -> None:
        """Teardown for each stage."""
        # clean up after fit or test
        # called on every process in DDP

    @property
    def hparams_initial(self) -> AttributeDict:
        """The collection of hyperparameters saved with `save_hyperparameters()`. It is read-only.

        The reason why we override is that we have some custom resolvers for `DictConfig`.
        Some resolved Python objects has not a primitive type, so that is not loggable without errors.
        Therefore, we need to unresolve it this time.
        """
        hp = super().hparams_initial
        for key, value in hp.items():
            if isinstance(value, DictConfig):
                # It should be unresolved to make it loggable
                hp[key] = OmegaConf.to_container(value, resolve=False)

        return hp

    def __reduce__(self):
        """Re-initialize object when unpickled."""
        return (self.__class__, (self.task, self.config))
