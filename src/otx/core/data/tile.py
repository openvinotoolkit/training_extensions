from __future__ import annotations
from typing import TYPE_CHECKING
import logging as log
from datumaro.plugins.tiling import Tile
from datumaro import Dataset as DmDataset
from otx.core.data.factory import OTXDatasetFactory
from otx.core.data.mem_cache import (
    MemCacheHandlerSingleton,
    parse_mem_cache_size_to_int,
)
from otx.core.data.module import OTXDataModule

if TYPE_CHECKING:
    from otx.core.data.entity.base import OTXBatchPredEntity


class OTXTileDataModule(OTXDataModule):
    def prepare_data(self) -> None:
        """Prepare data for each stage."""
        dataset = DmDataset.import_from(self.config.data_root, format=self.config.data_format)
        # TODO: convert tile size -> tile grid size
        # TODO: tile adapter
        dataset = dataset.transform(
            Tile,
            tile_size=self.config.tile_config.tile_size,
            overlap=self.config.tile_config.tile_overlap,
        )
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
                cfg_data_module=self.config,
            )
            log.info(f"Add name: {name}, self.subsets: {self.subsets}")

        mem_size = parse_mem_cache_size_to_int(self.config.mem_cache_size)
        mem_cache_mode = (
            "singleprocessing"
            if all(config.num_workers == 0 for config in config_mapping.values())
            else "multiprocessing"
        )

        self.mem_cache_handler = MemCacheHandlerSingleton.create(
            mode=mem_cache_mode,
            mem_size=mem_size,
        )

    def merge(self, predictions: OTXBatchPredEntity):
        """ Merge predictions from tiles to full image

        Args:
            predictions (_type_): _description_
        """
        # TODO: Implement this
        pass