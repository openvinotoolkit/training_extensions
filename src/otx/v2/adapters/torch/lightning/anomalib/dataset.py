"""Dataset Builder API for OTX anomalib adapter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from anomalib.data.base.datamodule import collate_fn
from omegaconf import DictConfig, OmegaConf

from otx.v2.adapters.torch.lightning.anomalib.modules.data.data import OTXAnomalyDataset
from otx.v2.adapters.torch.lightning.dataset import LightningDataset
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.entities.utils.dataset_utils import (
    contains_anomalous_images,
    split_local_global_dataset,
)
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.type_utils import str_to_subset_type, str_to_task_type

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DatumDataset
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import Dataset as TorchDataset
    from torch.utils.data import Sampler

SUBSET_LIST = ["train", "val", "test", "predict"]


@add_subset_dataloader(SUBSET_LIST)
class AnomalibDataset(LightningDataset):
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
            task (TaskType | str | None, optional): The type of anomaly task to perform.
                Defaults to None.
            train_type (TrainType | str | None, optional): The type of training to perform.
                Defaults to None.
            train_data_roots (str | None, optional): The path to the training data root directory.
                Defaults to None.
            train_ann_files (str | None, optional): The path to the training annotation file.
                Defaults to None.
            val_data_roots (str | None, optional): The path to the validation data root directory.
                Defaults to None.
            val_ann_files (str | None, optional): The path to the validation annotation file.
                Defaults to None.
            test_data_roots (str | None, optional): The path to the test data root directory.
                Defaults to None.
            test_ann_files (str | None, optional): The path to the test annotation file.
                Defaults to None.
            unlabeled_data_roots (str | None, optional): The path to the unlabeled data root directory.
                Defaults to None.
            unlabeled_file_list (str | None, optional): The path to the unlabeled file list.
                Defaults to None.
            data_format (str | None, optional): The format of the data. Defaults to "mvtec".

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

    def _build_dataset(
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

        config = OmegaConf.load(filename=config) if isinstance(config, str) else DictConfig({})

        config.dataset = {"transform_config": {"train": pipeline}, "image_size": [256, 256]}
        otx_dataset: DatumDataset = self.dataset_entity.get(str_to_subset_type(subset), None)
        if not otx_dataset or len(otx_dataset) < 1:
            return None

        if subset == "val":
            global_dataset, local_dataset = split_local_global_dataset(otx_dataset)
            otx_dataset = local_dataset if contains_anomalous_images(local_dataset) else global_dataset
        task_type = str_to_task_type(self.task) if isinstance(self.task, str) else self.task
        return OTXAnomalyDataset(config=config, dataset=otx_dataset, task_type=task_type)

    def _build_dataloader(
        self,
        dataset: OTXAnomalyDataset | None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Sampler | Iterable | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader | None:
        """Build a PyTorch DataLoader object from an OTXAnomalyDataset object.

        Args:
            dataset (OTXAnomalyDataset | None): The dataset to load.
            batch_size (int | None, optional): The batch size. Defaults to 1.
            num_workers (int | None, optional): The number of worker threads to use. Defaults to 0.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            pin_memory (bool, optional): Whether to pin memory. Defaults to False.
            drop_last (bool, optional): Whether to drop the last batch if it is smaller than the batch size.
                Defaults to False.
            sampler (Sampler | Iterable | None, optional): The sampler to use. Defaults to None.
            persistent_workers (bool, optional): Whether to keep the worker processes alive between data loading
                iterations. Defaults to False.
            **kwargs (Any): Takes additional parameters for the torch Dataloader.

        Returns:
            TorchDataLoader | None: The PyTorch DataLoader object.
        """
        return super()._build_dataloader(
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
