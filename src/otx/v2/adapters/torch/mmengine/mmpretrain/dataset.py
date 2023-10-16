"""OTX adapters.torch.mmengine.mmpretrain.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from functools import partial
from typing import Dict, Iterable, List, Optional, Union

import torch
from mmengine.dataset import default_collate, worker_init_fn
from mmengine.dist import get_dist_info
from mmengine.utils import digit_version
from mmpretrain.datasets import (
    build_dataset as mmpretrain_build_dataset,
)
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from otx.v2.adapters.torch.mmengine.mmpretrain.modules.datasets import (
    OTXClsDataset,
    OTXHierarchicalClsDataset,
    OTXMultilabelClsDataset,
    SelfSLDataset,
)
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.adapters.torch.modules.dataloaders import ComposedDL
from otx.v2.api.core.dataset import BaseDataset
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.type_utils import str_to_subset_type

SUBSET_LIST = ["train", "val", "test", "unlabeled"]


def get_default_pipeline(semisl: bool = False) -> Union[Dict, List]:
    """Returns the default pipeline for pretraining a model.

    Args:
        semisl (bool, optional): Whether to use a semi-supervised pipeline. Defaults to False.

    Returns:
        Union[Dict, List]: The default pipeline as a dictionary or list, depending on whether `semisl` is True or False.
    """
    # TODO: This is function for experiment // Need to remove this function
    default_pipeline = [
        {"type": "Resize", "scale": [224, 224]},
        {"type": "mmpretrain.PackInputs"},
    ]
    if semisl:
        strong_pipeline = [
            {"type": "OTXRandAugment", "num_aug": 8, "magnitude": 10},
        ]
        return {
            "train": default_pipeline,
            "unlabeled": [
                {"type": "Resize", "scale": [224, 224]},
                {"type": "PostAug", "keys": {"img_strong": strong_pipeline}},
                {"type": "mmpretrain.PackMultiKeyInputs", "input_key": "img", "multi_key": ["img_strong"]},
            ],
        }

    return default_pipeline


@add_subset_dataloader(SUBSET_LIST)
class Dataset(BaseDataset):
    """A class representing a dataset for pretraining a model."""

    def __init__(
        self,
        task: Optional[Union[TaskType, str]] = None,
        train_type: Optional[Union[TrainType, str]] = None,
        train_data_roots: Optional[str] = None,
        train_ann_files: Optional[str] = None,
        val_data_roots: Optional[str] = None,
        val_ann_files: Optional[str] = None,
        test_data_roots: Optional[str] = None,
        test_ann_files: Optional[str] = None,
        unlabeled_data_roots: Optional[str] = None,
        unlabeled_file_list: Optional[str] = None,
        data_format: Optional[str] = None,
    ) -> None:
        r"""MMPretrain's Dataset class.

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

    def _initialize(self) -> None:
        self.set_datumaro_adapters()  # Set self.dataset_entity & self.label_schema
        self.base_dataset = self._get_sub_task_dataset()
        self.initialize = True

    def _get_sub_task_dataset(self) -> TorchDataset:
        if self.train_type == TrainType.Selfsupervised:
            return SelfSLDataset
        len_group = len(self.label_schema.get_groups(False))
        len_labels = len(self.label_schema.get_labels(include_empty=False))
        if len_group > 1:
            if len_group == len_labels:
                return OTXMultilabelClsDataset
            return OTXHierarchicalClsDataset
        return OTXClsDataset

    def build_dataset(
        self,
        subset: str,
        pipeline: Optional[Union[list, dict]] = None,
        config: Optional[Union[str, dict, Config]] = None,
    ) -> Optional[TorchDataset]:
        """Builds a TorchDataset object for the given subset using the specified pipeline and configuration.

        Args:
            subset (str): The subset to build the dataset for.
            pipeline (Optional[Union[list, dict]]): The pipeline to use for the dataset.
                Defaults to None.
            config (Optional[Union[str, dict, Config]]): The configuration to use for the dataset.
                Defaults to None.

        Returns:
            Optional[TorchDataset]: The built TorchDataset object, or None if the dataset is empty.
        """
        if not self.initialize:
            self._initialize()

        if subset not in SUBSET_LIST:
            msg = f"{subset} is not supported subset"
            raise ValueError(msg)

        otx_dataset = self.dataset_entity.get_subset(str_to_subset_type(subset))
        labels = self.label_schema.get_labels(include_empty=False)
        if len(otx_dataset) < 1:
            return None
        # Case without config
        if config is None:
            _pipeline = pipeline if pipeline is not None else get_default_pipeline()
            dataset = self.base_dataset(otx_dataset=otx_dataset, labels=labels, pipeline=_pipeline)
            dataset.configs = {
                "type": str(self.base_dataset.__qualname__),
                "data_root": getattr(self, f"{subset}_data_roots"),
                "ann_file": getattr(self, f"{subset}_ann_files"),
                "data_prefix": "",
                "pipeline": _pipeline,
            }
            return dataset

        # Config Setting
        if isinstance(config, str):
            _config = Config.fromfile(filename=config)
        elif isinstance(config, dict):
            _config = Config(cfg_dict=config)
        else:
            _config = config
        # Case with Config
        dataset_config = _config.get("dataset", _config)
        init_config = dataset_config.copy()
        dataset_config["otx_dataset"] = otx_dataset
        dataset_config["labels"] = labels
        if pipeline is not None:
            dataset_config["pipeline"] = pipeline
        dataset_config.pop("data_roots", None)
        dataset_config.pop("ann_files", None)
        dataset_config.pop("file_list", None)
        dataset_config["_scope_"] = "mmpretrain"
        # Valid inputs
        if not dataset_config.get("type", False):
            dataset_config["type"] = self.base_dataset.__name__
        if not dataset_config.get("pipeline", False):
            dataset_config["pipeline"] = get_default_pipeline()
        dataset = mmpretrain_build_dataset(dataset_config)
        dataset.configs = init_config
        return dataset

    def build_dataloader(
        self,
        dataset: Optional[TorchDataset],
        batch_size: int = 2,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = True,
        sampler: Optional[Union[Sampler, Iterable, dict]] = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> Optional[TorchDataLoader]:
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
        rank, _ = get_dist_info()

        # Sampler
        seed = kwargs.get("seed", None)
        if isinstance(sampler, dict):
            pass
        if sampler is not None:
            shuffle = False

        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
        if digit_version(torch.__version__) >= digit_version("1.8.0"):
            kwargs["persistent_workers"] = persistent_workers

        dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(default_collate),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            **kwargs,
        )
        sampler_cfg = sampler if isinstance(sampler, dict) else {"type": f"{sampler.__class__.__qualname__}"}
        dataset_cfg = dataset.configs if hasattr(dataset, "configs") else dataset
        dataloader.configs = {
            "batch_size": batch_size,
            "sampler": sampler_cfg,
            "num_workers": num_workers,
            "collate_fn": {"type": "default_collate"},
            "pin_memory": pin_memory,
            "shuffle": shuffle,
            "dataset": dataset_cfg,
        }
        return dataloader

    def subset_dataloader(
        self,
        subset: str,
        pipeline: Optional[Union[dict, list]] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        config: Optional[Union[str, dict]] = None,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = True,
        sampler: Optional[Union[Sampler, Iterable, Dict]] = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader:
        r"""MMPretrain's Dataset.subset_dataloader.

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
        if isinstance(config, str):
            _config = Config.fromfile(filename=config)
        elif isinstance(config, dict):
            _config = Config(cfg_dict=config)
        elif config is None:
            _config = Config(cfg_dict={})
        else:
            _config = config
        dataloader_config = _config.get(f"{subset}_dataloader", None)
        subset_pipeline = pipeline
        if isinstance(subset_pipeline, dict):
            subset_pipeline = subset_pipeline[subset]
        subset_dataset = self.build_dataset(subset=subset, pipeline=subset_pipeline, config=dataloader_config)
        # TODO: argument update with configuration (config is not None case)
        if batch_size is None:
            batch_size = _config.get("batch_size", 2)
        if num_workers is None:
            num_workers = _config.get("num_workers", 0)

        # kwargs conflict
        unlabeled_batch_size = kwargs.pop("unlabeled_batch_size", _config.get("unlabeled_batch_size", batch_size))
        subset_dataloader = self.build_dataloader(
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
                default_pipeline = get_default_pipeline(semisl=True)
                if isinstance(default_pipeline, dict):
                    default_pipeline = default_pipeline.get("unlabeled", None)
                unlabeled_pipeline = pipeline.get("unlabeled", default_pipeline)
            unlabeled_dataset = self.build_dataset(subset="unlabeled", pipeline=unlabeled_pipeline, config=_config)
            unlabeled_dataloader = self.build_dataloader(
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
