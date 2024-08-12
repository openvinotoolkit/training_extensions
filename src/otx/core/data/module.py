# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LightningDataModule extension for OTX."""

from __future__ import annotations

import logging as log
from copy import deepcopy
from typing import TYPE_CHECKING, Iterable, Literal

import torch
from datumaro import Dataset as DmDataset
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler

from otx.core.config.data import TileConfig, UnlabeledDataConfig, VisualPromptingConfig
from otx.core.data.dataset.tile import OTXTileDatasetFactory
from otx.core.data.factory import OTXDatasetFactory
from otx.core.data.mem_cache import (
    MemCacheHandlerSingleton,
    parse_mem_cache_size_to_int,
)
from otx.core.data.pre_filtering import pre_filtering
from otx.core.data.utils import adapt_input_size_to_dataset, adapt_tile_config
from otx.core.types.device import DeviceType
from otx.core.types.image import ImageColorChannel
from otx.core.types.label import LabelInfo
from otx.core.types.task import OTXTaskType
from otx.core.utils.instantiators import instantiate_sampler
from otx.core.utils.utils import get_adaptive_num_workers

if TYPE_CHECKING:
    from lightning.pytorch.utilities.parsing import AttributeDict

    from otx.core.config.data import SubsetConfig
    from otx.core.data.dataset.base import OTXDataset


class OTXDataModule(LightningDataModule):
    """LightningDataModule extension for OTX pipeline.

    Args:
        input_size (int | tuple[int, int] | None, optional):
            Final image or video shape of data after data transformation. It'll be applied to all subset configs
            If it's not None. Defaults to None.
        adaptive_input_size (Literal["auto", "downscale"] | None, optional):
            The adaptive input size mode. If it's set, appropriate input size is found by analyzing dataset.
            "auto" can find both bigger and smaller input size than current input size and "downscale" uses only
            smaller size than default setting. Defaults to None.
        input_size_multiplier (int, optional):
            adaptive_input_size will finds multiple of input_size_multiplier value if it's set. It's usefull when
            a model requries multiple of specific value as input_size. Defaults to 1.
    """

    def __init__(  # noqa: PLR0913
        self,
        task: OTXTaskType,
        data_format: str,
        data_root: str,
        train_subset: SubsetConfig,
        val_subset: SubsetConfig,
        test_subset: SubsetConfig,
        unlabeled_subset: UnlabeledDataConfig = UnlabeledDataConfig(data_root=None),  # noqa: B008
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        vpm_config: VisualPromptingConfig = VisualPromptingConfig(),  # noqa: B008
        mem_cache_size: str = "1GB",
        mem_cache_img_max_size: tuple[int, int] | None = None,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        include_polygons: bool = False,
        ignore_index: int = 255,
        unannotated_items_ratio: float = 0.0,
        auto_num_workers: bool = False,
        device: DeviceType = DeviceType.auto,
        input_size: int | tuple[int, int] | None = None,
        adaptive_input_size: Literal["auto", "downscale"] | None = None,
        input_size_multiplier: int = 1,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.task = task
        self.data_format = data_format
        self.data_root = data_root

        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset
        self.unlabeled_subset = unlabeled_subset

        self.tile_config = tile_config
        self.vpm_config = vpm_config

        self.mem_cache_size = mem_cache_size
        self.mem_cache_img_max_size = mem_cache_img_max_size

        self.image_color_channel = image_color_channel
        self.stack_images = stack_images
        self.include_polygons = include_polygons
        self.ignore_index = ignore_index
        self.unannotated_items_ratio = unannotated_items_ratio

        self.auto_num_workers = auto_num_workers
        self.device = device

        self.subsets: dict[str, OTXDataset] = {}
        self.save_hyperparameters(ignore=["input_size"])

        # TODO (Jaeguk): This is workaround for a bug in Datumaro.
        # These lines should be removed after next datumaro release.
        # https://github.com/openvinotoolkit/datumaro/pull/1223/files
        from datumaro.plugins.data_formats.video import VIDEO_EXTENSIONS

        VIDEO_EXTENSIONS.append(".mp4")

        dataset = DmDataset.import_from(self.data_root, format=self.data_format)
        if self.task != "H_LABEL_CLS":
            dataset = pre_filtering(
                dataset,
                self.data_format,
                self.unannotated_items_ratio,
                ignore_index=self.ignore_index if self.task == "SEMANTIC_SEGMENTATION" else None,
            )

        unlabeled_dataset = None
        if self.unlabeled_subset.data_root is not None:
            # If unlabeled_subset's data_root is not None, use that folder as the Unlabeled dataset root.
            log.info(
                f"Unlabeled dataset is loaded from {self.unlabeled_subset.data_root}.",
            )
            unlabeled_dataset = DmDataset.import_from(
                self.unlabeled_subset.data_root,
                format=self.unlabeled_subset.data_format,
                subset=self.unlabeled_subset.subset_name,
            )

        if adaptive_input_size is not None:
            input_size = adapt_input_size_to_dataset(
                dataset,
                input_size,
                adaptive_input_size == "downscale",
                input_size_multiplier,
            )
        if input_size is not None:
            for subset_cfg in [train_subset, val_subset, test_subset, unlabeled_subset]:
                if subset_cfg.input_size is None:
                    subset_cfg.input_size = input_size
        self.input_size = input_size

        if self.tile_config.enable_tiler and self.tile_config.enable_adaptive_tiling:
            adapt_tile_config(self.tile_config, dataset=dataset)

        config_mapping = {
            self.train_subset.subset_name: self.train_subset,
            self.val_subset.subset_name: self.val_subset,
            self.test_subset.subset_name: self.test_subset,
        }

        if self.auto_num_workers:
            if self.device not in [DeviceType.gpu, DeviceType.auto]:
                log.warning(
                    "Only GPU device type support auto_num_workers. "
                    f"Current deveice type is {self.device!s}. auto_num_workers is skipped.",
                )
            elif (num_workers := get_adaptive_num_workers()) is not None:
                for subset_name, subset_config in config_mapping.items():
                    log.info(
                        f"num_workers of {subset_name} subset is changed : "
                        f"{subset_config.num_workers} -> {num_workers}",
                    )
                    subset_config.num_workers = num_workers

        mem_size = parse_mem_cache_size_to_int(mem_cache_size)
        mem_cache_mode = (
            "singleprocessing"
            if all(config.num_workers == 0 for config in config_mapping.values())
            else "multiprocessing"
        )
        mem_cache_handler = MemCacheHandlerSingleton.create(
            mode=mem_cache_mode,
            mem_size=mem_size,
        )

        label_infos: list[LabelInfo] = []
        for name, dm_subset in dataset.subsets().items():
            if name not in config_mapping:
                log.warning(f"{name} is not available. Skip it")
                continue

            dataset = OTXDatasetFactory.create(
                task=self.task,
                dm_subset=dm_subset.as_dataset(),
                cfg_subset=config_mapping[name],
                mem_cache_handler=mem_cache_handler,
                mem_cache_img_max_size=mem_cache_img_max_size,
                image_color_channel=image_color_channel,
                stack_images=stack_images,
                include_polygons=include_polygons,
                ignore_index=ignore_index,
                vpm_config=vpm_config,
            )

            if self.tile_config.enable_tiler:
                dataset = OTXTileDatasetFactory.create(
                    task=self.task,
                    dataset=dataset,
                    tile_config=self.tile_config,
                )
            self.subsets[name] = dataset

            label_infos += [self.subsets[name].label_info]
            log.info(f"Add name: {name}, self.subsets: {self.subsets}")

        if unlabeled_dataset is not None:
            name = self.unlabeled_subset.subset_name
            dm_subset = unlabeled_dataset.subsets()[name]

            if isinstance(self.unlabeled_subset.transforms, dict):
                # When applying multi-transforms to a single unlabeled dataset
                # This adds as many subsets as the number of keys in the transforms. The dataset is the same,
                # only the transforms are different.
                dm_subset = dm_subset.as_dataset()
                for transform_key, transforms in self.unlabeled_subset.transforms.items():
                    unlabeled_config = deepcopy(self.unlabeled_subset)
                    # TODO (harimkang): Revisit this with core.config.data.UnlabeledDataConfig.transforms.
                    unlabeled_config.transforms = transforms  # type: ignore[assignment]

                    unlabeled_dataset = OTXDatasetFactory.create(
                        task=self.task,
                        dm_subset=dm_subset,
                        cfg_subset=unlabeled_config,
                        mem_cache_handler=mem_cache_handler,
                        mem_cache_img_max_size=mem_cache_img_max_size,
                        image_color_channel=image_color_channel,
                        stack_images=stack_images,
                        include_polygons=include_polygons,
                        ignore_index=ignore_index,
                        vpm_config=vpm_config,
                    )
                    self.subsets[transform_key] = unlabeled_dataset
            else:
                unlabeled_dataset = OTXDatasetFactory.create(
                    task=self.task,
                    dm_subset=dm_subset.as_dataset(),
                    cfg_subset=self.unlabeled_subset,
                    mem_cache_handler=mem_cache_handler,
                    mem_cache_img_max_size=mem_cache_img_max_size,
                    image_color_channel=image_color_channel,
                    stack_images=stack_images,
                    include_polygons=include_polygons,
                    ignore_index=ignore_index,
                    vpm_config=vpm_config,
                )
                self.subsets[name] = unlabeled_dataset

        if self._is_meta_info_valid(label_infos) is False:
            msg = "All data meta infos of subsets should be the same."
            raise ValueError(msg)

        self.label_info = next(iter(label_infos))

    def _is_meta_info_valid(self, label_infos: list[LabelInfo]) -> bool:
        """Check whether there are mismatches in the metainfo for the all subsets."""
        return bool(all(label_info == label_infos[0] for label_info in label_infos))

    def _get_dataset(self, subset: str) -> OTXDataset:
        if (dataset := self.subsets.get(subset)) is None:
            msg = f"Dataset has no '{subset}'. Available subsets = {list(self.subsets.keys())}"
            raise KeyError(msg)
        return dataset

    def train_dataloader(self) -> Iterable:
        """Get train dataloader."""
        config = self.train_subset
        dataset = self._get_dataset(config.subset_name)
        sampler = instantiate_sampler(config.sampler, dataset=dataset, batch_size=config.batch_size)

        common_args = {
            "dataset": dataset,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": True,
            "collate_fn": dataset.collate_fn,
            "persistent_workers": config.num_workers > 0,
            "sampler": sampler,
            "shuffle": sampler is None,
        }

        tile_config = self.tile_config
        if tile_config.enable_tiler and tile_config.sampling_ratio < 1:
            num_samples = max(1, int(len(dataset) * tile_config.sampling_ratio))
            log.info(f"Using tiled sampling with {num_samples} samples")
            common_args.update(
                {
                    "shuffle": False,
                    "sampler": RandomSampler(dataset, num_samples=num_samples),
                },
            )
        dataloader: DataLoader = DataLoader(**common_args)
        if (unlabeled_dataloader := self.unlabeled_dataloader()) is not None:
            # Utilize the CombinedLoader provided by Lightning to bundle multiple dataloaders.
            # https://lightning.ai/docs/pytorch/stable/data/iterables.html
            from lightning.pytorch.utilities import CombinedLoader

            iterables = {
                "labeled": dataloader,
                **unlabeled_dataloader,
            }
            # CombinedLoader should always behave relative to the labeled dataloader.
            # if len(labeled_dataloader) < len(unlabeled_dataloader), the mode should be "min_size"
            # if len(labeled_dataloader) > len(unlabeled_dataloader), the mode should be "max_size_cycle"
            min_unlabeled_length = min(len(loader) for loader in unlabeled_dataloader.values())
            mode = "min_size" if len(dataloader) < min_unlabeled_length else "max_size_cycle"
            return CombinedLoader(iterables, mode=mode)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        """Get val dataloader."""
        config = self.val_subset
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
        config = self.test_subset
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
        config = self.test_subset
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

    def unlabeled_dataloader(self) -> dict[str, DataLoader] | None:
        """Returns a dictionary of unlabeled dataloaders.

        The method creates and returns dataloaders for unlabeled datasets based on the configuration settings.
        If the data root is not specified in the configuration, it returns None.

        Returns:
            dict[str, DataLoader] | None: A dictionary containing unlabeled dataloaders, where the keys are the names of
            the datasets and the values are the corresponding DataLoader objects.
        """
        config = self.unlabeled_subset
        if config.data_root is None:
            return None

        common_args = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": True,
            "persistent_workers": config.num_workers > 0,
        }

        dataloaders = {}
        if isinstance(config.transforms, dict):
            log.warning(f"Unlabeled dataset has multiple transforms : {list(config.transforms.keys())}")
            common_args["worker_init_fn"] = lambda _: torch.manual_seed(0)  # type: ignore[assignment]
            for key in config.transforms:
                dataset = self._get_dataset(key)
                # For unlabeled datasets using Multi-Transforms, must use generators and samplers
                # with the same seed to get the same data.
                generator = torch.Generator().manual_seed(0)
                sampler = instantiate_sampler(
                    config.sampler,
                    dataset=dataset,
                    batch_size=config.batch_size,
                    generator=generator,
                )

                dataloaders[key] = DataLoader(
                    dataset=dataset,
                    collate_fn=dataset.collate_fn,
                    sampler=sampler,
                    shuffle=False,
                    **common_args,
                )
        else:
            dataset = self._get_dataset(config.subset_name)
            sampler = instantiate_sampler(config.sampler, dataset=dataset, batch_size=config.batch_size)
            dataloaders[config.subset_name] = DataLoader(
                dataset=dataset,
                collate_fn=dataset.collate_fn,
                sampler=sampler,
                shuffle=sampler is None,
                **common_args,
            )

        return dataloaders

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
        return (
            self.__class__,
            (
                self.task,
                self.data_format,
                self.data_root,
                self.train_subset,
                self.val_subset,
                self.test_subset,
                self.unlabeled_subset,
                self.tile_config,
                self.vpm_config,
                self.mem_cache_size,
                self.mem_cache_img_max_size,
                self.image_color_channel,
                self.stack_images,
                self.include_polygons,
                self.ignore_index,
                self.unannotated_items_ratio,
                self.auto_num_workers,
                self.device,
                self.input_size,
            ),
        )
