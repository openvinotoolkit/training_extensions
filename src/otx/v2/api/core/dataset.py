"""OTX Core Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import os
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DatumDataset

from otx.v2.api.entities.label_schema import LabelSchemaEntity
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.auto_utils import configure_task_type, configure_train_type
from otx.v2.api.utils.type_utils import str_to_task_type, str_to_train_type

ADAPTERS = {
    TaskType.CLASSIFICATION: {
        "Incremental": {
            "module_name": "classification_dataset_adapter",
            "class": "ClassificationDatasetAdapter",
        },
        "Selfsupervised": {
            "module_name": "classification_dataset_adapter",
            "class": "SelfSLClassificationDatasetAdapter",
        },
    },
    TaskType.DETECTION: {
        "Incremental": {
            "module_name": "detection_dataset_adapter",
            "class": "DetectionDatasetAdapter",
        },
    },
    TaskType.ROTATED_DETECTION: {
        "Incremental": {
            "module_name": "detection_dataset_adapter",
            "class": "DetectionDatasetAdapter",
        },
    },
    TaskType.INSTANCE_SEGMENTATION: {
        "Incremental": {
            "module_name": "detection_dataset_adapter",
            "class": "DetectionDatasetAdapter",
        },
    },
    TaskType.SEGMENTATION: {
        "Incremental": {
            "module_name": "segmentation_dataset_adapter",
            "class": "SegmentationDatasetAdapter",
        },
        "Selfsupervised": {
            "module_name": "segmentation_dataset_adapter",
            "class": "SelfSLSegmentationDatasetAdapter",
        },
    },
    TaskType.ANOMALY_CLASSIFICATION: {
        "Incremental": {
            "module_name": "anomaly_dataset_adapter",
            "class": "AnomalyClassificationDatasetAdapter",
        },
    },
    TaskType.ANOMALY_DETECTION: {
        "Incremental": {
            "module_name": "anomaly_dataset_adapter",
            "class": "AnomalyDetectionDatasetAdapter",
        },
    },
    TaskType.ANOMALY_SEGMENTATION: {
        "Incremental": {
            "module_name": "anomaly_dataset_adapter",
            "class": "AnomalySegmentationDatasetAdapter",
        },
    },
}
if os.getenv("FEATURE_FLAGS_OTX_ACTION_TASKS", "0") == "1":
    ADAPTERS.update(
        {
            TaskType.ACTION_CLASSIFICATION: {
                "Incremental": {
                    "module_name": "action_dataset_adapter",
                    "class": "ActionClassificationDatasetAdapter",
                },
            },
            TaskType.ACTION_DETECTION: {
                "Incremental": {
                    "module_name": "action_dataset_adapter",
                    "class": "ActionDetectionDatasetAdapter",
                },
            },
        },
    )
if os.getenv("FEATURE_FLAGS_OTX_VISUAL_PROMPTING_TASKS", "0") == "1":
    ADAPTERS.update(
        {
            TaskType.VISUAL_PROMPTING: {
                "Incremental": {
                    "module_name": "visual_prompting_dataset_adapter",
                    "class": "VisualPromptingDatasetAdapter",
                },
            },
        },
    )


class BaseDatasetAdapter:
    """Base class for dataset adapters."""

    def __init__(
        self,
        task_type: TaskType,
        train_data_roots: str | None = None,
        train_ann_files: str | None = None,
        val_data_roots: str | None = None,
        val_ann_files: str | None = None,
        test_data_roots: str | None = None,
        test_ann_files: str | None = None,
        unlabeled_data_roots: str | None = None,
        unlabeled_file_list: str | None = None,
        cache_config: dict | None = None,
        encryption_key: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize a new instance of the DatasetAdapter class.

        Args:
            task_type (TaskType): The type of task that the dataset is intended for.
            train_data_roots (Optional[str], optional): The root directory or directories containing the training data.
                Defaults to None.
            train_ann_files (Optional[str], optional): The annotation file or files for the training data.
                Defaults to None.
            val_data_roots (Optional[str], optional): The root directory or directories containing the validation data.
                Defaults to None.
            val_ann_files (Optional[str], optional): The annotation file or files for the validation data.
                Defaults to None.
            test_data_roots (Optional[str], optional): The root directory or directories containing the test data.
                Defaults to None.
            test_ann_files (Optional[str], optional): The annotation file or files for the test data. Defaults to None.
            unlabeled_data_roots (Optional[str], optional): The root directory
                or directories containing the unlabeled data. Defaults to None.
            unlabeled_file_list (Optional[str], optional): The file containing a list of unlabeled data files.
                Defaults to None.
            cache_config (Optional[dict], optional): The configuration for the dataset cache. Defaults to None.
            encryption_key (Optional[str], optional): The encryption key to use for the dataset. Defaults to None.
            **kwargs: Additional keyword arguments.
        """

    @abstractmethod
    def get_otx_dataset(self) -> dict[Subset, DatumDataset]:
        """Get Dict[Subset, DatumDataset]."""
        raise NotImplementedError

    def get_label_schema(self) -> LabelSchemaEntity:
        """Get Label Schema."""
        raise NotImplementedError


def get_dataset_adapter(
    task_type: TaskType,
    train_type: TrainType,
    train_data_roots: str | None = None,
    train_ann_files: str | None = None,
    val_data_roots: str | None = None,
    val_ann_files: str | None = None,
    test_data_roots: str | None = None,
    test_ann_files: str | None = None,
    unlabeled_data_roots: str | None = None,
    unlabeled_file_list: str | None = None,
    **kwargs,
) -> BaseDatasetAdapter:
    """Return a dataset class by task type.

    Args:
        task_type: A task type such as ANOMALY_CLASSIFICATION, ANOMALY_DETECTION, ANOMALY_SEGMENTATION,
            CLASSIFICATION, INSTANCE_SEGMENTATION, DETECTION, CLASSIFICATION, ROTATED_DETECTION, SEGMENTATION.
        train_type: train type such as Incremental and Selfsupervised.
            Selfsupervised is only supported for SEGMENTATION.
        train_data_roots: the path of data root for training data
        train_ann_files: the path of annotation file for training data
        val_data_roots: the path of data root for validation data
        val_ann_files: the path of annotation file for validation data
        test_data_roots: the path of data root for test data
        test_ann_files: the path of annotation file for test data
        unlabeled_data_roots: the path of data root for unlabeled data
        unlabeled_file_list: the path of unlabeled file list
        **kwargs: optional kwargs
    """
    train_type_to_be_called = str(
        train_type if train_type == TrainType.Selfsupervised.value else TrainType.Incremental.value,
    )
    module_root = "otx.v2.adapters.datumaro.adapter."
    module = importlib.import_module(module_root + ADAPTERS[task_type][train_type_to_be_called]["module_name"])
    return getattr(module, ADAPTERS[task_type][train_type_to_be_called]["class"])(
        task_type=task_type,
        train_data_roots=train_data_roots,
        train_ann_files=train_ann_files,
        val_data_roots=val_data_roots,
        val_ann_files=val_ann_files,
        test_data_roots=test_data_roots,
        test_ann_files=test_ann_files,
        unlabeled_data_roots=unlabeled_data_roots,
        unlabeled_file_list=unlabeled_file_list,
        **kwargs,
    )


class BaseDataset:
    """Base class for providing the underlying Dataset API for each framework.

    Example:
    >>> dataset = Dataset(
        train_data_roots="train/data/roots/",
        val_data_roots="val/data/roots/",
        test_data_roots="test/data/roots/",
    )
    """

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
        """BaseDataset, Classes that provide the underlying Dataset API for each framework.

        Args:
            task (Optional[Union[TaskType, str]], optional): The task type of the dataset want to load.
                Defaults to None.
            train_type (Optional[Union[TrainType, str]], optional): The train type of the dataset want to load.
                Defaults to None.
            train_data_roots (Optional[str], optional): The root address of the dataset to be used for training.
                Defaults to None.
            train_ann_files (Optional[str], optional): Location of the annotation file for the dataset
                to be used for training. Defaults to None.
            val_data_roots (Optional[str], optional): The root address of the dataset to be used for validation.
                Defaults to None.
            val_ann_files (Optional[str], optional): Location of the annotation file for the dataset
                to be used for validation. Defaults to None.
            test_data_roots (Optional[str], optional): The root address of the dataset to be used for testing.
                Defaults to None.
            test_ann_files (Optional[str], optional): Location of the annotation file for the dataset
                to be used for testing. Defaults to None.
            unlabeled_data_roots (Optional[str], optional): The root address of the unlabeled dataset
                to be used for training. Defaults to None.
            unlabeled_file_list (Optional[str], optional): The file where the list of unlabeled images is declared.
                Defaults to None.
            data_format (Optional[str], optional): The format of the dataset. Defaults to None.
        """
        self._train_data_roots = train_data_roots
        self._train_ann_files = train_ann_files
        self._val_data_roots = val_data_roots
        self._val_ann_files = val_ann_files
        self._test_data_roots = test_data_roots
        self._test_ann_files = test_ann_files
        self._unlabeled_data_roots = unlabeled_data_roots
        self._unlabeled_file_list = unlabeled_file_list

        self.task = task
        self.train_type = train_type
        self.data_format = data_format
        self.initialize = False

    def set_datumaro_adapters(self, data_roots: str | None = None) -> None:
        """Functions that provide the ability to load datasets from datumaro.

        If the train-type dataset for a particular task is supported by the datumaro adapter,
        this provides the ability to easily load the dataset using this function.
        This can also be detected automatically by using Task-Type, Train-Type with data_roots.

        Args:
            data_roots (Optional[str], optional): The root address of the dataset to be used for Task
                or Train-Type auto detection.

        How to use:
            1) Create a BaseDataset
            2) Call this function
            3) This will set
                - self.dataset_adapter (DatumaroDatasetAdapter)
                - self.dataset_entity (Dict[Subset, DatumDataset])
                - self.label_schema (LabelSchemaEntity) so that can use it.
            4) Use the above attributes to complete the subset_dataloader function.
        """
        # Task & Train-Type Auto Detection
        if data_roots is None:
            data_roots = self.train_data_roots
        if self.task is None:
            self.task, self.data_format = configure_task_type(data_roots, self.data_format)
        if self.train_type is None:
            self.train_type = configure_train_type(data_roots, self.unlabeled_data_roots)

        # String to TaskType & TrainType
        self.task = str_to_task_type(self.task) if isinstance(self.task, str) else self.task
        self.train_type = str_to_train_type(self.train_type) if isinstance(self.train_type, str) else self.train_type

        self.dataset_adapter = get_dataset_adapter(
            task_type=self.task,
            train_type=self.train_type,
            train_data_roots=self.train_data_roots,
            train_ann_files=self.train_ann_files,
            val_data_roots=self.val_data_roots,
            val_ann_files=self.val_ann_files,
            test_data_roots=self.test_data_roots,
            test_ann_files=self.test_ann_files,
            unlabeled_data_roots=self.unlabeled_data_roots,
            unlabeled_file_list=self.unlabeled_file_list,
        )
        self.label_schema: LabelSchemaEntity = self.dataset_adapter.get_label_schema()
        self.dataset_entity: dict[Subset, DatumDataset] = self.dataset_adapter.get_otx_dataset()

    @abstractmethod
    def subset_dataloader(  # noqa: ANN201
        self,
        subset: str,
        pipeline: dict | list | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        **kwargs,
    ):
        """BaseDataset's subset_dataloader function.

        Functions for calling dataloader for each subset of the dataset class.
        It needs to be implemented to use a decorator called @add_subset_dataloader.

        Args:
            subset (str): Subset of dataloader.
            pipeline (Optional[Union[List, Dict]], optional): The data pipe to apply to that dataset. Defaults to None.
            batch_size (Optional[int], optional): Batch size of this dataloader. Defaults to None.
            num_workers (Optional[int], optional): Number of workers for this dataloader. Defaults to None.
            **kwargs (optional): Additional keyword arguments to be passed to the dataloader.

        Returns:
            DataLoader: The dataloader for the specified subset.

        Example:
        >>> train_dataloader = dataset.train_dataloader()
        Dataloader()
        >>> train_dataloader = dataset.train_dataloader(batch_size=4, num_workers=2)
        Dataloader()
        >>> val_dataloader = dataset.val_dataloader()
        Dataloader()
        >>> test_dataloader = dataset.test_dataloader()
        Dataloader()

        How to use:
            1) Implement this function to return the dataloader of each subset.
            2) Declare the @add_subset_dataloader(subset_list) decorator with subset_list in the class.
            3) BaseDataset.{subset}_dataloader() will call that function.
                BaseDataset.{subset}_dataloader() == BaseDataset.subset_dataloader(subset)
        """

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset.

        :return: An integer representing the number of classes in the dataset.
        """
        raise NotImplementedError

    @property
    def train_data_roots(self) -> str | None:
        """Returns the root directory for training data, if set.

        :return: The root directory for training data, or None if not set.
        """
        return self._train_data_roots

    @train_data_roots.setter
    def train_data_roots(self, path: str) -> None:
        self._train_data_roots = path
        self.initialize = False

    @property
    def train_ann_files(self) -> str | None:
        """Returns the path to the training annotation files for this dataset.

        Returns:
            Optional[str]: The path to the training annotation files, or None if not set.
        """
        return self._train_ann_files

    @train_ann_files.setter
    def train_ann_files(self, path: str) -> None:
        self._train_ann_files = path
        self.initialize = False

    @property
    def val_data_roots(self) -> str | None:
        """Returns the root directory for validation data, if set.

        :return: The root directory for validation data, or None if not set.
        """
        return self._val_data_roots

    @val_data_roots.setter
    def val_data_roots(self, path: str) -> None:
        self._val_data_roots = path
        self.initialize = False

    @property
    def val_ann_files(self) -> str | None:
        """Returns the path to the validation annotation files for this dataset.

        Returns:
            Optional[str]: The path to the validation annotation files, or None if not set.
        """
        return self._val_ann_files

    @val_ann_files.setter
    def val_ann_files(self, path: str) -> None:
        self._val_ann_files = path
        self.initialize = False

    @property
    def test_data_roots(self) -> str | None:
        """Returns the root directory for testing data, if set.

        :return: The root directory for testing data, or None if not set.
        """
        return self._test_data_roots

    @test_data_roots.setter
    def test_data_roots(self, path: str) -> None:
        self._test_data_roots = path
        self.initialize = False

    @property
    def test_ann_files(self) -> str | None:
        """Returns the path to the test annotation files for this dataset.

        Returns:
            Optional[str]: The path to the test annotation files, or None if not set.
        """
        return self._test_ann_files

    @test_ann_files.setter
    def test_ann_files(self, path: str) -> None:
        self._test_ann_files = path
        self.initialize = False

    @property
    def unlabeled_data_roots(self) -> str | None:
        """Returns the root directory for unlabeled data, if set.

        :return: The root directory for unlabeled data, or None if not set.
        """
        return self._unlabeled_data_roots

    @unlabeled_data_roots.setter
    def unlabeled_data_roots(self, path: str) -> None:
        self._unlabeled_data_roots = path
        self.initialize = False

    @property
    def unlabeled_file_list(self) -> str | None:
        """Returns the path to the unlabeled image list for this dataset.

        Returns:
            Optional[str]: The path to the unlabeled image list, or None if not set.
        """
        return self._unlabeled_file_list

    @unlabeled_file_list.setter
    def unlabeled_file_list(self, path: str) -> None:
        self._unlabeled_file_list = path
        self.initialize = False
