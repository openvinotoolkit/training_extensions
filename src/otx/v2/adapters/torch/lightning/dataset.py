"""OTX adapters.torch.lightning.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from otx.v2.adapters.torch.lightning.modules.datasets import OTXVisualPromptingDataset
from otx.v2.adapters.torch.lightning.modules.datasets.pipelines import collate_fn
from otx.v2.api.core.dataset import BaseDataset
from otx.v2.api.entities.task_type import TaskType
from otx.v2.api.utils import set_tuple_constructor
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.type_utils import str_to_subset_type

SUBSET_LIST = ["train", "val", "test"]


@add_subset_dataloader(SUBSET_LIST)
class LightningDataset(BaseDataset):
    """A class representing a dataset for Lightning."""

    def _initialize(self) -> None:
        self.set_datumaro_adapters()
        self.initialize = True

    def build_dataset(
        self,
        subset: str,
        pipeline: dict | list | None = None,  # transform_config
        config: str | (DictConfig | dict) | None = None,
    ) -> TorchDataset | None:
        """Build a TorchDataset for the given subset using the specified pipeline and configuration.

        Args:
            subset (str): The subset to build the dataset for.
            pipeline (dict | list | None, optional): The pipeline to use for data transformation.
            config (str | (DictConfig | dict) | None, optional): The configuration to use for the dataset.

        Returns:
            TorchDataset | None: The built TorchDataset, or None if the dataset is empty.
        """
        if not self.initialize:
            self._initialize()

        # NOTE: It needs to be refactored in a more general way.
        if self.task in (TaskType.VISUAL_PROMPTING, "visual_prompting"):
            config = OmegaConf.load(filename=config) if isinstance(config, str) else DictConfig({})
            config = config.get("dataset", config)
            image_size = config.get("image_size", 1024)
            normalize = config.get("normalize", {})
            mean = normalize.get("mean", [123.675, 116.28, 103.53])
            std = normalize.get("std", [58.395, 57.12, 57.375])
            offset_bbox = config.get("offset_bbox", 20)

            if subset == "predict":
                otx_dataset = self.dataset_entity
            else:
                otx_dataset = self.dataset_entity.get_subset(str_to_subset_type(subset))
            if len(otx_dataset) < 1:
                return None

            return OTXVisualPromptingDataset(
                dataset=otx_dataset,
                image_size=image_size,
                mean=mean,
                std=std,
                offset_bbox=offset_bbox,
                pipeline=pipeline,
            )
        raise NotImplementedError

    def build_dataloader(
        self,
        dataset: TorchDataset | None,
        batch_size: int | None = 1,
        num_workers: int | None = 0,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Sampler | Iterable | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader | None:
        """Build a PyTorch DataLoader object from an TorchDataset object.

        Args:
            dataset (Optional[TorchDataset]): The dataset to load.
            batch_size (Optional[int], optional): The batch size. Defaults to 1.
            num_workers (Optional[int], optional): The number of worker threads to use. Defaults to 0.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            pin_memory (bool, optional): Whether to pin memory. Defaults to False.
            drop_last (bool, optional): Whether to drop the last batch if it is smaller than the batch size.
                Defaults to False.
            sampler (Optional[Union[Sampler, Iterable]], optional): The sampler to use. Defaults to None.
            persistent_workers (bool, optional): Whether to keep the worker processes alive between data loading
                iterations. Defaults to False.
            **kwargs (Any): Takes additional parameters for the torch Dataloader.

        Returns:
            Optional[TorchDataLoader]: The PyTorch DataLoader object.
        """
        if dataset is None:
            return None
        if sampler is not None:
            shuffle = False

        # Currently, copy from mm's
        input_collate_fn = kwargs.pop("collate_fn", collate_fn)
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=input_collate_fn,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs,
        )

    def subset_dataloader(
        self,
        subset: str,
        pipeline: dict | list | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        config: str | dict | None = None,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Sampler | Iterable | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader:
        """Return a PyTorch DataLoader for the specified subset of the dataset.

        Args:
            subset (str): The subset of the dataset to load. Must be one of "train", "val", "test", or "predict".
            pipeline (dict | list | None, optional): A pipeline of transformations to apply to the data.
            batch_size (Optional[int]): The batch size to use for the DataLoader. If not provided, will be
                determined by the configuration file or default to 1.
            num_workers (Optional[int]): The number of worker processes to use for loading the data. If not
                provided, will be determined by the configuration file or default to 0.
            config (Optional[Union[str, dict]]): The configuration file or dictionary to use for determining
                the batch size and number of workers. If a string is provided, it will be treated as a path to
                a YAML file.
            shuffle (bool): Whether to shuffle the data before loading.
            pin_memory (bool): Whether to pin the memory of the loaded data to improve performance.
            drop_last (bool): Whether to drop the last batch if it is smaller than the specified batch size.
            sampler (Optional[Union[Sampler, Iterable]]): A sampler to use for loading the data.
            persistent_workers (bool): Whether to keep the worker processes alive between batches to improve
                performance.
            **kwargs: Additional keyword arguments to pass to the DataLoader constructor.

        Returns:
            A PyTorch DataLoader for the specified subset of the dataset.
        """
        _config: dict = {}
        if isinstance(config, str):
            set_tuple_constructor()
            with Path(config).open() as f:
                _config = yaml.safe_load(f)
        elif config is not None:
            _config = config

        dataset = self.build_dataset(subset=subset, pipeline=pipeline, config=_config)
        if batch_size is None:
            batch_size = _config.get("batch_size", 1)
        if num_workers is None:
            num_workers = _config.get("num_workers", 0)

        return self.build_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            **kwargs,
        )

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset.

        If the dataset has not been initialized, this method will call the _initialize method to initialize it.

        Returns:
            int: The number of classes in the dataset.
        """
        if not self.initialize:
            self._initialize()
        return len(self.label_schema.get_labels(include_empty=False))
