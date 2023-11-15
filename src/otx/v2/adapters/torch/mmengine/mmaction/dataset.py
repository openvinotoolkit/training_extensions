"""OTX adapters.torch.mmengine.mmaction.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Iterable

import torch
from mmengine.dataset import pseudo_collate, worker_init_fn
from mmengine.dist import get_dist_info
from mmengine.utils import digit_version
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from otx.v2.adapters.torch.mmengine.dataset import MMXDataset
from otx.v2.adapters.torch.mmengine.mmaction.modules.datasets import (
    OTXActionClsDataset,
)
from otx.v2.adapters.torch.mmengine.mmaction.registry import MMActionRegistry
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.decorators import add_subset_dataloader

SUBSET_LIST = ["train", "val", "test", "predict"]


def get_default_pipeline(subset: str) -> list:
    """Returns the default pipeline for mmaction model.

    Args:
        subset (str): Subset of default pipeline

    Returns:
        List: The default pipeline as a dictionary or list, depending on whether `semisl` is True or False.
    """
    if subset not in SUBSET_LIST:
        msg = f"{subset} is not supported subset"
        raise ValueError(msg)

    default_pipeline = {
        "train": [
            {
                "type": "SampleFrames",
                "clip_len": 8,
                "frame_interval": 4,
                "num_clips": 1,
            },
            {"type": "OTXRawFrameDecode"},
            {"type": "Resize", "scale": (-1, 256)},
            {"type": "FormatShape", "input_format": "NCTHW"},
            {"type": "PackActionInputs"},
        ],
        "val": [
            {
                "type": "SampleFrames",
                "clip_len": 8,
                "frame_interval": 4,
                "num_clips": 1,
                "test_mode": True,
            },
            {"type": "OTXRawFrameDecode"},
            {"type": "Resize", "scale": (-1, 256)},
            {"type": "CenterCrop", "crop_size": 224},
            {"type": "FormatShape", "input_format": "NCTHW"},
            {"type": "PackActionInputs"},
        ],
        "test": [
            {
                "type": "SampleFrames",
                "clip_len": 8,
                "frame_interval": 4,
                "num_clips": 1,
                "test_mode": True,
            },
            {"type": "OTXRawFrameDecode"},
            {"type": "Resize", "scale": (-1, 256)},
            {"type": "CenterCrop", "crop_size": 224},
            {"type": "FormatShape", "input_format": "NCTHW"},
            {"type": "PackActionInputs"},
        ],
        "predict":[
            {
                "type": "SampleFrames",
                "clip_len": 8,
                "frame_interval": 4,
                "num_clips": 1,
                "test_mode": True,
            },
            {"type": "RawFrameDecode"},
            {"type": "Resize", "scale": (-1, 256)},
            {"type": "CenterCrop", "crop_size": 224},
            {"type": "FormatShape", "input_format": "NCTHW"},
            {"type": "PackActionInputs"},
        ],
    }
    return default_pipeline[subset]



@add_subset_dataloader(SUBSET_LIST)
class MMActionDataset(MMXDataset):
    """A class representing a dataset for mmaction model."""

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
        r"""MMAction's Dataset class.

        Args:
            task (TaskType | str | None, optional): The task type of the dataset want to load.
                Defaults to None.
            train_type (TrainType | str | None, optional): The train type of the dataset want to load.
                Defaults to None.
            train_data_roots (str | None, optional): The root address of the dataset to be used for training.
                Defaults to None.
            train_ann_files (str | None, optional): Location of the annotation file for the dataset
                to be used for training. Defaults to None.
            val_data_roots (str | None, optional): The root address of the dataset
                to be used for validation. Defaults to None.
            val_ann_files (str | None, optional): Location of the annotation file for the dataset
                to be used for validation. Defaults to None.
            test_data_roots (str | None, optional): The root address of the dataset
                to be used for testing. Defaults to None.
            test_ann_files (str | None, optional): Location of the annotation file for the dataset
                to be used for testing. Defaults to None.
            unlabeled_data_roots (str | None, optional): The root address of the unlabeled dataset
                to be used for training. Defaults to None.
            unlabeled_file_list (str | None, optional): The file where the list of unlabeled images is declared.
                Defaults to None.
            data_format (str | None, optional): The format of the dataset. Defaults to None.
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
        self.scope = "mmaction"
        self.dataset_registry = MMActionRegistry().get("dataset")

    def _initialize(self) -> None:
        self.set_datumaro_adapters()  # Set self.dataset_entity & self.label_schema
        self.base_dataset = self._get_sub_task_dataset()
        self.initialize = True

    def _get_sub_task_dataset(self) -> TorchDataset:
        # ruff: noqa: TD002, TD003, FIX002
        #TODO: will be added the OTXActionDetDataset
        return OTXActionClsDataset

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
            TorchDataset | None: The built TorchDataset object, or None if the dataset is empty.
        """
        if pipeline is None:
            pipeline = get_default_pipeline(subset)
        return super()._build_dataset(subset, pipeline, config)

    def _build_dataloader(
        self,
        dataset: TorchDataset | None,
        batch_size: int | None = 2,
        num_workers: int | None = 0,
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
        rank, _ = get_dist_info()

        # Sampler
        seed = kwargs.get("seed", None)

        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
        if digit_version(torch.__version__) >= digit_version("1.8.0"):
            kwargs["persistent_workers"] = persistent_workers
        
        shuffle = False
        dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(pseudo_collate),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            **kwargs,
        )
        dataset_cfg = dataset.configs if hasattr(dataset, "configs") else dataset
        dataloader.configs = {
            "batch_size": batch_size,
            "sampler": sampler,
            "num_workers": num_workers,
            "collate_fn": {"type": "pseudo_collate"},
            "pin_memory": pin_memory,
            "shuffle": shuffle,
            "dataset": dataset_cfg,
        }
        return dataloader
