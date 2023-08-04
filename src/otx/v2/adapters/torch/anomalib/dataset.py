"""OTX adapters.torch.anomalib.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import yaml
from anomalib.data.base.datamodule import collate_fn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from otx.v2.adapters.torch.anomalib.modules.data.data import OTXAnomalyDataset
from otx.v2.api.core.dataset import BaseDataset
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.entities.utils.dataset_utils import (
    contains_anomalous_images,
    split_local_global_dataset,
)
from otx.v2.api.utils import set_tuple_constructor
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.type_utils import str_to_subset_type, str_to_task_type

if TYPE_CHECKING:
    import albumentations as al

SUBSET_LIST = ["train", "val", "test", "predict"]


@add_subset_dataloader(SUBSET_LIST)
class Dataset(BaseDataset):
    """A class representing a dataset for Anomaly tasks."""

    def __init__(
        self,
        task: TaskType | str | None = None,
        train_type: TrainType | str | None = None,
        train_data_roots: str | None = None,
        train_ann_files: str | None = None,
        val_data_roots: str | None = None,
        val_ann_files: str | None = None,
        test_data_roots: str | None = None,
        test_ann_files: str | None = None,
        unlabeled_data_roots: str | None = None,
        unlabeled_file_list: str | None = None,
        data_format: str | None = "mvtec",
    ) -> None:
        """Initialize a Dataset object for anomaly tasks.

        Args:
            task (Optional[Union[TaskType, str]]): The type of anomaly task to perform.
                Defaults to None.
            train_type (Optional[Union[TrainType, str]]): The type of training to perform.
                Defaults to None.
            train_data_roots (Optional[str]): The path to the training data root directory.
                Defaults to None.
            train_ann_files (Optional[str]): The path to the training annotation file.
                Defaults to None.
            val_data_roots (Optional[str]): The path to the validation data root directory.
                Defaults to None.
            val_ann_files (Optional[str]): The path to the validation annotation file.
                Defaults to None.
            test_data_roots (Optional[str]): The path to the test data root directory.
                Defaults to None.
            test_ann_files (Optional[str]): The path to the test annotation file.
                Defaults to None.
            unlabeled_data_roots (Optional[str]): The path to the unlabeled data root directory.
                Defaults to None.
            unlabeled_file_list (Optional[str]): The path to the unlabeled file list.
                Defaults to None.
            data_format (Optional[str]): The format of the data. Defaults to "mvtec".

        Returns:
            None
        """
        if isinstance(task, str) and not task.startswith("anomaly"):
            task = "anomaly_" + task
        super().__init__(
            task,
            train_type,
            train_data_roots,
            train_ann_files,
            val_data_roots,
            val_ann_files,
            test_data_roots,
            test_ann_files,
            unlabeled_data_roots,
            unlabeled_file_list,
            data_format=data_format,
        )

    def _initialize(self) -> None:
        self.set_datumaro_adapters()
        self.initialize = True

    def build_dataset(
        self,
        subset: str,
        pipeline: str | al.Compose | None = None,  # transform_config
        config: str | (DictConfig | dict) | None = None,
    ) -> TorchDataset | None:
        """Build a TorchDataset for the given subset using the specified pipeline and configuration.

        Args:
            subset (str): The subset to build the dataset for.
            pipeline (Optional[Union[str, al.Compose]]): The pipeline to use for data transformation.
            config (Optional[Union[str, DictConfig, dict]]): The configuration to use for the dataset.

        Returns:
            Optional[TorchDataset]: The built TorchDataset, or None if the dataset is empty.
        """
        if not self.initialize:
            self._initialize()

        config = OmegaConf.load(filename=config) if isinstance(config, str) else DictConfig({})

        config.dataset = {"transform_config": {"train": pipeline}, "image_size": [256, 256]}
        otx_dataset = self.dataset_entity.get_subset(str_to_subset_type(subset))
        if len(otx_dataset) < 1:
            return None

        if subset == "val":
            global_dataset, local_dataset = split_local_global_dataset(otx_dataset)
            otx_dataset = local_dataset if contains_anomalous_images(local_dataset) else global_dataset
        task_type = str_to_task_type(self.task) if isinstance(self.task, str) else self.task
        return OTXAnomalyDataset(config=config, dataset=otx_dataset, task_type=task_type)

    def build_dataloader(
        self,
        dataset: OTXAnomalyDataset | None,
        batch_size: int | None = 1,
        num_workers: int | None = 0,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Sampler | Iterable | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader | None:
        """Build a PyTorch DataLoader object from an OTXAnomalyDataset object.

        Args:
            dataset (Optional[OTXAnomalyDataset]): The dataset to load.
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
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
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
        batch_size: int | None = 1,
        num_workers: int | None = 0,
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
            pipeline (Optional[Union[dict, list]]): A pipeline of transformations to apply to the data.
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
        if subset == "predict":
            pass
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
