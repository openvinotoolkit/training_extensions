"""OTX adapters.torch.mmengine.mmdet.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset

from otx.v2.adapters.torch.mmengine.dataset import MMXDataset
from otx.v2.adapters.torch.mmengine.mmdet.modules.datasets import (
    OTXDetDataset,
)
from otx.v2.adapters.torch.mmengine.mmdet.registry import MMDetRegistry
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.decorators import add_subset_dataloader

SUBSET_LIST = ["train", "val", "test", "unlabeled"]


def get_default_pipeline(semisl: bool = False) -> dict | list:
    """Returns the default pipeline for training a model.

    Args:
        semisl (bool, optional): Whether to use a semi-supervised pipeline. Defaults to False.

    Returns:
        Union[Dict, List]: The default pipeline as a dictionary or list, depending on whether `semisl` is True or False.
    """
    err_msg = "SemiSL is not implemented in MMDet task."
    if semisl:
        raise NotImplementedError(err_msg)

    return [
        {
            "type": "LoadResizeDataFromOTXDataset",
            "load_ann_cfg": {"type": "LoadAnnotationFromOTXDataset", "with_bbox": True},
            "resize_cfg": {"type": "Resize", "scale": (640, 640), "keep_ratio": True, "downscale_only": True},
            "enable_memcache": True,
        },
        {"type": "Resize", "scale": [640, 640], "keep_ratio": True},
        {"type": "Pad"},
        {"type": "RandomFlip", "prob": 0.5},
        {"type": "PackInputs"},
    ]


@add_subset_dataloader(SUBSET_LIST)
class MMDetDataset(MMXDataset):
    """A class representing a dataset for training a model."""

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
        r"""MMDet's Dataset class.

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
        self.scope = "mmdet"
        self.dataset_registry = MMDetRegistry().get("dataset")

    def _get_sub_task_dataset(self) -> TorchDataset:
        return OTXDetDataset

    def _build_dataset(
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
        if pipeline is None:
            semisl = subset == "unlabeled"
            pipeline = get_default_pipeline(semisl=semisl)
        return super()._build_dataset(subset, pipeline, config)
