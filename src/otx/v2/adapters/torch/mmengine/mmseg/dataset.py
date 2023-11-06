"""OTX adapters.torch.mmengine.mmpretrain.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Iterable

import torch
from mmengine.dataset import pseudo_collate, worker_init_fn
from mmengine.dist import get_dist_info
from mmengine.utils import digit_version
from mmseg.registry import DATA_SAMPLERS, DATASETS
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from otx.v2.adapters.torch.mmengine.mmseg.modules.datasets import OTXSegDataset
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.core.dataset import BaseDataset
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.type_utils import str_to_subset_type

SUBSET_LIST = ["train", "val", "test", "unlabeled"]


def get_default_pipeline(subset: str) -> list:
    """Returns the default data processing pipeline for the given subset.

    Args:
        subset (str): The subset of the dataset. Can be "train", "val", or "test".

    Returns:
        list: The list of processing steps to be applied to the data.
    """
    if subset == "train":
        return [
            {"type": "LoadImageFromOTXDataset"},
            {"type": "LoadAnnotationFromOTXDataset", "_scope_": "mmseg"},
            {"type": "RandomResize", "scale": (544, 544), "ratio_range": (0.5, 2.0)},
            {"type": "RandomCrop", "crop_size": (512, 512), "cat_max_ratio": 0.75},
            {"type": "RandomFlip", "prob": 0.5, "direction": "horizontal"},
            {"type": "PackSegInputs", "_scope_": "mmseg"},
        ]
    if subset in ("val", "test"):
        return [
            {"type": "LoadImageFromOTXDataset"},
            {"type": "Resize", "scale": (544, 544)},
            {"type": "LoadAnnotationFromOTXDataset", "_scope_": "mmseg"},
            {"type": "PackSegInputs", "_scope_": "mmseg"},
        ]
    msg = "Not supported subset"
    raise NotImplementedError(msg)


@add_subset_dataloader(SUBSET_LIST)
class Dataset(BaseDataset):
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
        r"""MMSeg's Dataset class.

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
        self.initialize = True

    def build_dataset(
        self,
        subset: str,
        pipeline: list | dict | None = None,
        config: str | (dict | Config) | None = None,
    ) -> TorchDataset | None:
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
            _pipeline = pipeline if pipeline is not None else get_default_pipeline(subset=subset)
            dataset = OTXSegDataset(otx_dataset=otx_dataset, labels=labels, pipeline=_pipeline)
            dataset.configs = {
                "type": str(OTXSegDataset.__qualname__),
                "data_root": getattr(self, f"{subset}_data_roots"),
                "ann_file": getattr(self, f"{subset}_ann_files"),
                "data_prefix": "",
                "pipeline": _pipeline,
            }
            return dataset

        # TODO (Eugene): load dataset from config
        # CVS-124394

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
            dataset_config["type"] = OTXSegDataset.__name__
        if not dataset_config.get("pipeline", False):
            dataset_config["pipeline"] = get_default_pipeline(subset=subset)
        dataset = DATASETS.build(dataset_config)
        dataset.configs = init_config
        return dataset

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
        # TODO (Eugene): almost identical as mmengine.runner. Need to rethink about the design.
        # CVS-124394

        if dataset is None:
            return None
        rank, _ = get_dist_info()

        # mmengine build dataloader in runner where we build in dataset.
        # So, I'm basically copying things from runner.build_dataloader to here
        # which does not make sense. why do we not use dataloader_cfg and build dataloader
        # in runner.build_dataloader?
        seed = kwargs.get("seed", 0)

        # build sampler
        if sampler is None:
            sampler = DATA_SAMPLERS.build(
                {"type": "DefaultSampler", "shuffle": shuffle},
                default_args={"dataset": dataset, "seed": seed},
            )

        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
        if digit_version(torch.__version__) >= digit_version("1.8.0"):
            kwargs["persistent_workers"] = persistent_workers

        dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(pseudo_collate),
            pin_memory=pin_memory,
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
            "collate_fn": {"type": "pseudo_collate"},
            "pin_memory": pin_memory,
            "shuffle": shuffle,
            "dataset": dataset_cfg,
        }
        return dataloader

    def subset_dataloader(
        self,
        subset: str,
        pipeline: dict | list | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        config: str | dict | None = None,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = True,
        sampler: Sampler | (Iterable | dict) | None = None,
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
        # TODO (eugene): direct pass config
        # CVS-124394
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
        if batch_size is None:
            batch_size = _config.get("batch_size", 2)
        if num_workers is None:
            num_workers = _config.get("num_workers", 0)

        return self.build_dataloader(
            dataset=subset_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle if subset == "train" else False,
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
