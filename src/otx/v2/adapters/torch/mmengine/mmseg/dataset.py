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
from mmseg.registry import DATA_SAMPLERS
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from otx.v2.adapters.torch.mmengine.dataset import MMXDataset
from otx.v2.adapters.torch.mmengine.mmseg.modules.datasets import OTXSegDataset
from otx.v2.adapters.torch.mmengine.mmseg.registry import MMSegmentationRegistry
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.decorators import add_subset_dataloader

SUBSET_LIST = ["train", "val", "test", "unlabeled"]


def get_default_pipeline(subset: str) -> list:
    """Returns the default data processing pipeline for the given subset.

    Args:
        subset (str): The subset of the dataset. Can be "train", "val", or "test".

    Returns:
        list: The list of processing steps to be applied to the data.
    """
    # TODO (Eugene): Implement LoadResizeDataFromOTXDataset in second phase.
    # CVS-124394
    if subset == "train":
        return [
            {"type": "LoadImageFromOTXDataset"},
            {"type": "LoadAnnotationFromOTXDataset", "_scope_": "mmseg"},
            {"type": "RandomResize", "scale": (544, 544), "ratio_range": (0.5, 2.0)},
            {"type": "RandomCrop", "crop_size": (512, 512), "cat_max_ratio": 0.75, "_scope_": "mmseg"},
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
class MMSegDataset(MMXDataset):
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
        """MMSeg's Dataset class.

        Args:
            task (TaskType | str, optional): The task type of the dataset want to load.
                Defaults to None.
            train_type (TaskType | str, optional): The train type of the dataset want to load.
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
        self.scope = "mmseg"
        self.dataset_registry = MMSegmentationRegistry().get("dataset")

    def _get_sub_task_dataset(self) -> TorchDataset:
        return OTXSegDataset

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
            dataset (TorchDataset | None): The dataset to load.
            batch_size (int): The batch size to use.
            num_workers (int): The number of worker processes to use for data loading.
            shuffle (bool): Whether to shuffle the data.
            pin_memory (bool): Whether to pin memory for faster GPU transfer.
            drop_last (bool): Whether to drop the last incomplete batch.
            sampler (Sampler | Iterable | Dict | None): The sampler to use for data loading.
            persistent_workers (bool): Whether to keep the worker processes alive between iterations.
            **kwargs: Additional arguments to pass to the DataLoader constructor.

        Returns:
            TorchDataLoader | None: The DataLoader for the given dataset.
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
        dataset_config = config.get("dataset", config) if config is not None else {}
        if pipeline is None and "pipeline" not in dataset_config:
            pipeline = get_default_pipeline(subset=subset)
        return super()._build_dataset(subset, pipeline, config)

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset.

        If the dataset has not been initialized, this method will first initialize it.

        Returns:
            The number of classes in the dataset.
        """
        # TODO (Eugene): add test cases
        # CVS-124394
        if not self.initialize:
            self._initialize()
        label_names = [lbs.name for lbs in self.label_schema.get_labels(include_empty=False)]
        if "background" in label_names:
            return len(label_names)
        return len(label_names) + 1
