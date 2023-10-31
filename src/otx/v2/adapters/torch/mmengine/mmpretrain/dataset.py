"""OTX adapters.torch.mmengine.mmpretrain.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.v2.adapters.torch.mmengine.dataset import MMXDataset
from otx.v2.adapters.torch.mmengine.mmpretrain.modules.datasets import (
    OTXClsDataset,
    OTXHierarchicalClsDataset,
    OTXMultilabelClsDataset,
    SelfSLDataset,
)
from otx.v2.adapters.torch.mmengine.mmpretrain.registry import MMPretrainRegistry
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.decorators import add_subset_dataloader

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset

SUBSET_LIST = ["train", "val", "test", "unlabeled"]


def get_default_pipeline(semisl: bool = False) -> dict | list:
    """Returns the default pipeline for pretraining a model.

    Args:
        semisl (bool, optional): Whether to use a semi-supervised pipeline. Defaults to False.

    Returns:
        Union[Dict, List]: The default pipeline as a dictionary or list, depending on whether `semisl` is True or False.
    """
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
class MMPretrainDataset(MMXDataset):
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
        self.scope = "mmpretrain"
        self.dataset_registry = MMPretrainRegistry().get("dataset")

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
        return super().build_dataset(subset, pipeline, config)
