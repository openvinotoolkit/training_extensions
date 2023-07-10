"""OTX Core Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
from typing import Dict, List, Optional, Union

from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.label_schema import LabelSchemaEntity
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
        }
    },
    TaskType.ROTATED_DETECTION: {
        "Incremental": {
            "module_name": "detection_dataset_adapter",
            "class": "DetectionDatasetAdapter",
        }
    },
    TaskType.INSTANCE_SEGMENTATION: {
        "Incremental": {
            "module_name": "detection_dataset_adapter",
            "class": "DetectionDatasetAdapter",
        }
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
        }
    },
    TaskType.ANOMALY_DETECTION: {
        "Incremental": {
            "module_name": "anomaly_dataset_adapter",
            "class": "AnomalyDetectionDatasetAdapter",
        }
    },
    TaskType.ANOMALY_SEGMENTATION: {
        "Incremental": {
            "module_name": "anomaly_dataset_adapter",
            "class": "AnomalySegmentationDatasetAdapter",
        }
    },
}
if os.getenv("FEATURE_FLAGS_OTX_ACTION_TASKS", "0") == "1":
    ADAPTERS.update(
        {
            TaskType.ACTION_CLASSIFICATION: {
                "Incremental": {
                    "module_name": "action_dataset_adapter",
                    "class": "ActionClassificationDatasetAdapter",
                }
            },
            TaskType.ACTION_DETECTION: {
                "Incremental": {
                    "module_name": "action_dataset_adapter",
                    "class": "ActionDetectionDatasetAdapter",
                }
            },
        }
    )
# TODO: update to real template
if os.getenv("FEATURE_FLAGS_OTX_VISUAL_PROMPTING_TASKS", "0") == "1":
    ADAPTERS.update(
        {
            TaskType.VISUAL_PROMPTING: {
                "Incremental": {
                    "module_name": "visual_prompting_dataset_adapter",
                    "class": "VisualPromptingDatasetAdapter",
                }
            },
        }
    )


def get_dataset_adapter(
    task_type: TaskType,
    train_type: TrainType,
    train_data_roots: str = None,
    train_ann_files: str = None,
    val_data_roots: str = None,
    val_ann_files: str = None,
    test_data_roots: str = None,
    test_ann_files: str = None,
    unlabeled_data_roots: str = None,
    unlabeled_file_list: str = None,
    **kwargs,
):
    """Returns a dataset class by task type.

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
        kwargs: optional kwargs
    """

    train_type_to_be_called = str(
        train_type if train_type == TrainType.Selfsupervised.value else TrainType.Incremental.value
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
        """Classes for loading a Dataset from dataset paths.

        Args:
            task (Optional[Union[TaskType, str]], optional): The task type of the dataset want to load. Defaults to None.
            train_type (Optional[Union[TrainType, str]], optional): The train type of the dataset want to load. Defaults to None.
            train_data_roots (Optional[str], optional): The root address of the dataset to be used for training. Defaults to None.
            train_ann_files (Optional[str], optional): Location of the annotation file for the dataset to be used for training. Defaults to None.
            val_data_roots (Optional[str], optional): _description_. Defaults to None.
            val_ann_files (Optional[str], optional): Location of the annotation file for the dataset to be used for validation. Defaults to None.
            test_data_roots (Optional[str], optional): _description_. Defaults to None.
            test_ann_files (Optional[str], optional): Location of the annotation file for the dataset to be used for testing. Defaults to None.
            unlabeled_data_roots (Optional[str], optional): The root address of the unlabeled dataset to be used for training. Defaults to None.
            unlabeled_file_list (Optional[str], optional): The file where the list of unlabeled images is declared. Defaults to None.
            data_format (Optional[str], optional): The format of the dataset. Defaults to None.
        """
        self.task = task
        self.train_type = train_type
        self.train_data_roots = train_data_roots
        self.train_ann_files = train_ann_files
        self.val_data_roots = val_data_roots
        self.val_ann_files = val_ann_files
        self.test_data_roots = test_data_roots
        self.test_ann_files = test_ann_files
        self.unlabeled_data_roots = unlabeled_data_roots
        self.unlabeled_file_list = unlabeled_file_list
        self.data_format = data_format

    def set_datumaro_adapters(self, data_roots: Optional[str] = None):
        """Functions that provide the ability to load datasets from datumaro.

        If the train-type dataset for a particular task is supported by the datumaro adapter,
        this provides the ability to easily load the dataset using this function.
        This can also be detected automatically by using Task-Type, Train-Type with data_roots.

        Args:
            data_roots (Optional[str], optional): The root address of the dataset to be used for Task or Train-Type auto detection.

        How to use:
            1) Create a BaseDataset
            2) Call this function
            3) This will set
                - self.dataset_adapter (BaseDatasetAdapter)
                - self.dataset_entity (DatasetEntity)
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
        self.dataset_entity: DatasetEntity = self.dataset_adapter.get_otx_dataset()
        self.label_schema: LabelSchemaEntity = self.dataset_adapter.get_label_schema()

    def subset_dataloader(self, subset: str, *args, **kwargs):
        """BaseDataset's subset_dataloader function.

        Functions for calling dataloader for each subset of the dataset class.
        It needs to be implemented to use a decorator called @add_subset_dataloader.

        Args:
            subset (str): Subset of a dataset.

        Raises:
            NotImplementedError: Must be implemented to use the subset dataloader.

        How to use:
            1) Implement this function to return the dataloader of each subset.
            2) Declare the @add_subset_dataloader(subset_list) decorator with subset_list in the class.
            3) BaseDataset.{subset}_dataloader() will call that function.
                BaseDataset.{subset}_dataloader() == BaseDataset.subset_dataloader(subset)
        """
        raise NotImplementedError()

    @property
    def num_classes(self) -> int:
        raise NotImplementedError()
