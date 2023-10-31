"""OTX adapters.torch.mmengine.mmpretrain.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterable

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from otx.v2.api.core.dataset import BaseDataset
from otx.v2.api.utils.decorators import add_subset_dataloader

SUBSET_LIST = ["train", "val", "test"]


@add_subset_dataloader(SUBSET_LIST)
class TorchBaseDataset(BaseDataset):
    """A class representing a dataset for pretraining a model."""

    def _initialize(self) -> None:
        self.set_datumaro_adapters()  # Set self.dataset_entity & self.label_schema
        self.initialize = True

    def build_dataset(
        self,
        subset: str,
        pipeline: list | dict | None = None,
        config: str | dict | None = None,
    ) -> TorchDataset | None:
        """Builds a TorchDataset object for the given subset using the specified pipeline and configuration.

        Args:
            subset (str): The subset to build the dataset for.
            pipeline (Optional[Union[list, dict]]): The pipeline to use for the dataset.
                Defaults to None.
            config (Optional[Union[str, dict]]): The configuration to use for the dataset.
                Defaults to None.

        Returns:
            Optional[TorchDataset]: The built TorchDataset object, or None if the dataset is empty.
        """


    def build_dataloader(
        self,
        dataset: TorchDataset | None,
        batch_size: int = 2,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = True,
        sampler: Sampler | (Iterable | dict) | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader | None:
        """Builds a PyTorch DataLoader for the given dataset.

        Args:
            dataset (Optional[TorchDataset]): The dataset to load.
            batch_size (int): The batch size to use.
            num_workers (int): The number of worker processes to use for data loading.
            shuffle (bool): Whether to shuffle the data.
            pin_memory (bool): Whether to pin memory for faster GPU transfer.
            drop_last (bool): Whether to drop the last incomplete batch.
            sampler (Optional[Union[Sampler, Iterable, Dict]]): The sampler to use for data loading.
            persistent_workers (bool): Whether to keep the worker processes alive between iterations.
            **kwargs: Additional arguments to pass to the DataLoader constructor.

        Returns:
            Optional[TorchDataLoader]: The DataLoader for the given dataset.
        """
        if dataset is None:
            return None

        # Sampler
        if isinstance(sampler, dict):
            pass
        if sampler is not None:
            shuffle = False

        dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs,
        )
        sampler_cfg = sampler if isinstance(sampler, dict) else {"type": f"{sampler.__class__.__qualname__}"}
        dataset_cfg = dataset.configs if hasattr(dataset, "configs") else dataset
        dataloader.configs = {
            "batch_size": batch_size,
            "sampler": sampler_cfg,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "shuffle": shuffle,
            "persistent_workers": persistent_workers,
            "dataset": dataset_cfg,
            **kwargs,
        }
        return dataloader

    def subset_dataloader(
        self,
        subset: str,
        pipeline: dict | list | None = None,
        batch_size: int = 2,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = True,
        sampler: Sampler | (Iterable | dict) | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader:
        r"""Torch Based Dataset.subset_dataloader.

        Args:
            subset (str): Enter an available subset of that dataset.
            pipeline (Optional[Union[list, dict]], optional):
                Dataset Pipeline. Defaults to None.
            batch_size (Optional[int], optional): How many samples per batch to load. Defaults to None.
            num_workers (Optional[int], optional): How many subprocesses to use for data loading.
                ``0`` means that the data will be loaded in the main process. Defaults to None.
            config (Optional[Union[str, dict]], optional): Path to configuration file or Config.
                Defaults to None.
            shuffle (bool, optional): Set to ``True`` to have the data reshuffled at every epoch. Defaults to True.
            pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
                into device/CUDA pinned memory before returning them.  If your data elements
                are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
                see the example below. Defaults to False.
            drop_last (bool, optional): value for whether to drop the last data when the batch is not divided up.
                Defaults to False.
            sampler (Optional[Union[Sampler, Iterable, Dict]], optional): Defines the strategy to draw
                samples from the dataset. Can be any ``Iterable`` with ``__len__``
                implemented. If specified, :attr:`shuffle` must not be specified.. Defaults to None.
            persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
                the worker processes after a dataset has been consumed once. This allows to
                maintain the workers `Dataset` instances alive. Defaults to False.
            **kwargs (Any): Additional arguments to pass to the DataLoader constructor.

        Returns:
            torch.utils.data.DataLoader: Returns a subset of dataLoader.
        """
        # Config Setting
        subset_pipeline = pipeline
        if isinstance(subset_pipeline, dict):
            subset_pipeline = subset_pipeline[subset]
        subset_dataset = self.build_dataset(subset=subset, pipeline=subset_pipeline)

        # kwargs conflict
        return self.build_dataloader(
            dataset=subset_dataset,
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
        """Returns the number of classes in the dataset.

        If the dataset has not been initialized, this method will first initialize it.

        Returns:
            The number of classes in the dataset.
        """
        if not self.initialize:
            self._initialize()
        return len(self.label_schema.get_labels(include_empty=False))
