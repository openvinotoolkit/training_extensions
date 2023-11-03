"""OTX adapters.torch.mmengine.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import yaml
from mmengine.dataset import default_collate, worker_init_fn
from mmengine.dist import get_dist_info

from otx.v2.adapters.torch.dataset import BaseTorchDataset
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.adapters.torch.mmengine.registry import MMEngineRegistry
from otx.v2.adapters.torch.modules.dataloaders import ComposedDL
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils import set_tuple_constructor
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.type_utils import str_to_subset_type

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DatumDataset
    from mmengine.registry import Registry
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import Dataset as TorchDataset
    from torch.utils.data import Sampler

SUBSET_LIST = ["train", "val", "test"]


@add_subset_dataloader(SUBSET_LIST)
class MMXDataset(BaseTorchDataset):
    """A class representing a dataset for pretraining a model."""

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
        data_format: str | None = None,
    ) -> None:
        r"""MMEngine's Dataset class.

        Args:
            task (Optional[Union[TaskType, str]], optional): The task type of the dataset want to load.
                Defaults to None.
            train_type (Optional[Union[TrainType, str]], optional): The train type of the dataset want to load.
                Defaults to None.
            train_data_roots (Optional[str], optional): The root address of the dataset to be used for training.
                Defaults to None.
            train_ann_files (Optional[str], optional): Location of the annotation file for the dataset
                to be used for training. Defaults to None.
            val_data_roots (Optional[str], optional): The root address of the dataset
                to be used for validation. Defaults to None.
            val_ann_files (Optional[str], optional): Location of the annotation file for the dataset
                to be used for validation. Defaults to None.
            test_data_roots (Optional[str], optional): The root address of the dataset
                to be used for testing. Defaults to None.
            test_ann_files (Optional[str], optional): Location of the annotation file for the dataset
                to be used for testing. Defaults to None.
            unlabeled_data_roots (Optional[str], optional): The root address of the unlabeled dataset
                to be used for training. Defaults to None.
            unlabeled_file_list (Optional[str], optional): The file where the list of unlabeled images is declared.
                Defaults to None.
            data_format (Optional[str], optional): The format of the dataset. Defaults to None.

        Example:
        >>> MMXDataset(train_data_roots="dataset/train", val_data_roots="dataset/val")
        MMXDataset with train/val subsets
        """
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
            data_format,
        )
        self.scope = "mmengine"
        self.dataset_registry: Registry = MMEngineRegistry().get("dataset")

    def _initialize(self) -> None:
        self.set_datumaro_adapters()  # Set self.dataset_entity & self.label_schema
        self.base_dataset = self._get_sub_task_dataset()
        self.initialize = True

    def _get_sub_task_dataset(self) -> TorchDataset:
        """Returns a TorchDataset object containing the data for the current sub-task.

        Returns:
            TorchDataset: A TorchDataset object containing the data for the current sub-task.
        """
        raise NotImplementedError

    def _build_dataset(
        self,
        subset: str,
        pipeline: list | None = None,
        config: dict | None = None,
    ) -> TorchDataset | None:
        """Builds a TorchDataset object for the given subset using the specified pipeline and configuration.

        Args:
            subset (str): The subset to build the dataset for.
            pipeline (list | None, optional): The pipeline to use for the dataset.
                Defaults to None.
            config (dict | None, optional): The configuration to use for the dataset.
                Defaults to None.

        Returns:
            Optional[TorchDataset]: The built TorchDataset object, or None if the dataset is empty.
        """
        if not self.initialize:
            self._initialize()

        if subset not in SUBSET_LIST:
            msg = f"{subset} is not supported subset"
            raise ValueError(msg)

        otx_dataset: DatumDataset = self.dataset_entity.get(str_to_subset_type(subset))
        labels = self.label_schema.get_labels(include_empty=False)
        if not otx_dataset or len(otx_dataset) < 1:
            return None

        # Case without config
        if config is None:
            dataset = self.base_dataset(otx_dataset=otx_dataset, labels=labels, pipeline=pipeline)
            dataset.configs = {
                "type": str(self.base_dataset.__qualname__),
                "data_root": getattr(self, f"{subset}_data_roots"),
                "ann_file": getattr(self, f"{subset}_ann_files"),
                "data_prefix": "",
                "pipeline": pipeline,
            }
            return dataset

        # Config Setting
        _config = Config(cfg_dict=config)
        dataset_config = _config.get("dataset", _config)
        init_config = dataset_config.copy()
        dataset_config["otx_dataset"] = otx_dataset
        dataset_config["labels"] = labels
        if pipeline is not None:
            dataset_config["pipeline"] = pipeline
        dataset_config.pop("data_roots", None)
        dataset_config.pop("ann_files", None)
        dataset_config.pop("file_list", None)
        dataset_config["_scope_"] = self.scope

        if not _config.get("type", False):
            _config["type"] = self.base_dataset.__name__
        dataset = self.dataset_registry.build(_config)
        dataset.configs = init_config
        return dataset

    def _build_dataloader(
        self,
        dataset: TorchDataset | None,
        batch_size: int | None = 2,
        num_workers: int | None = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
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
        rank, _ = get_dist_info()

        # Sampler
        seed = kwargs.get("seed", None)

        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
        kwargs["worker_init_fn"] = init_fn
        kwargs["collate_fn"] = partial(default_collate)

        return super()._build_dataloader(
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

    def subset_dataloader(
        self,
        subset: str,
        pipeline: dict[str, list] | list | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        config: str | dict | None = None,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Sampler | (Iterable | dict) | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader:
        r"""MMEngine's Dataset.subset_dataloader.

        This is a method named ``subset_dataloader`` within a class.
        This method is designed to create a PyTorch DataLoader for a specific subset of a dataset with mmX framework.
        The method returns a TorchDataLoader object, which is a PyTorch DataLoader configured
        according to the provided parameters. This DataLoader can then be used to iterate over the specified subset of
        the dataset in the specified batch size, potentially with multiple worker processes and with the data optionally
        reshuffled at every epoch.
        It also provides more versatility through the ``add_subset_dataloader`` decorator function.
        This minimizes duplication of code and provides a more intuitive function.
        Can see a detailed example of this in Example section.

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

        Example:
        >>> dataset.subset_dataloader(subset="train")
        torch.utils.data.Dataloader()
        >>> dataset.train_dataloader()
        torch.utils.data.Dataloader()
        """
        # Config Setting
        _config: dict = {}
        if isinstance(config, str):
            set_tuple_constructor()
            with Path(config).open() as f:
                _config = yaml.safe_load(f)
        elif config is not None:
            _config = config

        subset_pipeline = pipeline
        if isinstance(subset_pipeline, dict):
            subset_pipeline = subset_pipeline[subset]
        subset_dataset = self._build_dataset(subset=subset, pipeline=subset_pipeline, config=_config)
        if batch_size is None:
            batch_size = _config.get("batch_size", 2)
        if num_workers is None:
            num_workers = _config.get("num_workers", 0)

        # kwargs conflict
        unlabeled_batch_size = kwargs.pop("unlabeled_batch_size", _config.get("unlabeled_batch_size", batch_size))
        subset_dataloader = self._build_dataloader(
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
        if subset == "train" and self.train_type == TrainType.Semisupervised:
            unlabeled_pipeline = None
            if isinstance(pipeline, dict):
                unlabeled_pipeline = pipeline.get("unlabeled", None)
            unlabeled_dataset = self._build_dataset(subset="unlabeled", pipeline=unlabeled_pipeline, config=_config)
            unlabeled_dataloader = self._build_dataloader(
                dataset=unlabeled_dataset,
                batch_size=unlabeled_batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=pin_memory,
                drop_last=drop_last,
                sampler=sampler,
                persistent_workers=persistent_workers,
                **kwargs,
            )
            return ComposedDL([subset_dataloader, unlabeled_dataloader])
        return subset_dataloader
