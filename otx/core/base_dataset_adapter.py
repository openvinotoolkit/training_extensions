"""Base Class for Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member, too-many-return-statements, unused-argument

import abc
from abc import abstractmethod
from typing import Any, Dict, Tuple, Union

import datumaro
from datumaro.components.annotation import AnnotationType as DatumaroAnnotationType
from datumaro.components.dataset import Dataset as DatumaroDataset

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.label import LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.subset import Subset

def get_dataset_adapter(task_type):
    """Returns a dataset class by task type.
    Args:
        task_type: A task type such as ANOMALY_CLASSIFICATION, ANOMALY_DETECTION, ANOMALY_SEGMENTATION,
        CLASSIFICATION, INSTANCE_SEGMENTATION, DETECTION, CLASSIFICATION, ROTATED_DETECTION, SEGMENTATION.
    """
    if task_type == TaskType.CLASSIFICATION:
        from .classification_dataset_adapter import ClassificationDatasetAdapter

        return ClassificationDatasetAdapter(task_type=task_type)

    if task_type in [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION]:
        from .detection_dataset_adapter import DetectionDatasetAdapter

        return DetectionDatasetAdapter(task_type=task_type)

    if task_type == TaskType.SEGMENTATION:
        from .segmentation_dataset_adapter import SegmentationDatasetAdapter

        return SegmentationDatasetAdapter(task_type=task_type)

    if task_type == TaskType.ACTION_CLASSIFICATION:
        from .action_dataset_adapter import ActionClassificationDatasetAdapter

        return ActionClassificationDatasetAdapter(task_type=task_type)

    if task_type == TaskType.ANOMALY_CLASSIFICATION:
        from .anomaly_dataset_adapter import AnomalyClassificationDatasetAdapter

        return AnomalyClassificationDatasetAdapter(task_type=task_type)

    if task_type == TaskType.ANOMALY_DETECTION:
        from .anomaly_dataset_adapter import AnomalyDetectionDatasetAdapter

        return AnomalyDetectionDatasetAdapter(task_type=task_type)

    if task_type == TaskType.ANOMALY_SEGMENTATION:
        from .anomaly_dataset_adapter import AnomalySegmentationDatasetAdapter

        return AnomalySegmentationDatasetAdapter(task_type=task_type)

    # TODO: Need to implement
    # if task_type == TaskType.ACTION_DETECTION:
    #    from .action_dataset_adapter import ActionDetectionDatasetAdapter
    #
    #    return ActionDetectionDatasetAdapter(task_type=task_type)
    # if task_type == TaskType.ROTATED_DETECTION:
    #    from .rotated_detection.dataset import RotatedDetectionDataset
    #
    #    return RotatedDetectionDataset

    raise ValueError(f"Invalid task type: {task_type}")


class BaseDatasetAdapter(metaclass=abc.ABCMeta):
    """Base dataset adapter for all of downstream tasks to use Datumaro

    Mainly, BaseDatasetAdapter detect and import the dataset by using the function implemented in Datumaro.
    And it could prepare common variable, function (EmptyLabelSchema, LabelSchema, ..) commonly consumed under all tasks

    """

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.domain = task_type.domain
        self.data_type = None # type: Any
        self.dataset = None  # type: Any
        self.is_train_phase = None # type: bool

    def import_dataset(
        self,
        train_data_roots: str = None,
        val_data_roots: str = None,
        test_data_roots: str = None,
        unlabeled_data_roots: str = None,
    ) -> Dict[Subset, DatumaroDataset]:
        """Import dataset by using Datumaro.import_from() method.

        Args:
            train_data_roots (str): Path for training data
            train_ann_files (str): Path for training annotation data
            val_data_roots (str): Path for validation data
            val_ann_files (str): Path for validation annotation data
            test_data_roots (str): Path for test data
            test_ann_files (str): Path for test annotation data
            unlabeled_data_roots (str): Path for unlabeled data
            unlabeled_file_lists (str): Path for unlabeled file list

        Returns:
            DatumaroDataset: Datumaro Dataset
        """
        self.dataset = {}
        
        # Construct dataset for training, validation, testing, unlabeled
        if train_data_roots:
            # Find self.data_type and task_type
            data_type_candidates = self._detect_dataset_format(path=train_data_roots)
            self.data_type = self._select_data_type(data_type_candidates)
            
            datumaro_dataset = DatumaroDataset.import_from(train_data_roots, format=self.data_type)

            # Prepare subsets by using Datumaro dataset
            for k, v in datumaro_dataset.subsets().items():
                if "train" in k or "default" in k:
                    self.dataset[Subset.TRAINING] = v
                elif "val" in k:
                    self.dataset[Subset.VALIDATION] = v
            self.is_train_phase = True

            # If validation is manually defined --> set the validation data according to user's input
            if val_data_roots:
                val_data_candidates = self._detect_dataset_format(path=val_data_roots)
                val_data_type = self._select_data_type(val_data_candidates)
                self.dataset[Subset.VALIDATION] = DatumaroDataset.import_from(val_data_roots, format=val_data_type)

            if Subset.VALIDATION not in self.dataset:
                # TODO: auto_split
                pass
        
        if test_data_roots:
            test_data_candidates = self._detect_dataset_format(path=test_data_roots)
            test_data_type = self._select_data_type(test_data_candidates)
            self.dataset[Subset.TESTING] = DatumaroDataset.import_from(test_data_roots, format=test_data_type)
            self.is_train_phase = False

        if unlabeled_data_roots is not None:
            self.dataset[Subset.UNLABELED] = DatumaroDataset.import_from(unlabeled_data_roots, format="image_dir")

        return self.dataset

    @abstractmethod
    def convert_to_otx_format(self, datumaro_dataset: dict) -> Tuple[DatasetEntity, LabelSchemaEntity]:
        """Convert DatumaroDataset to the DatasetEntity.
        Args:
            datumaro_dataset (dict): A Dictionary that includes subset dataset(DatasetEntity)
        Returns:
            DatasetEntity:
        """
        raise NotImplementedError

    def _detect_dataset_format(self, path: str) -> str:
        """Detect dataset format (ImageNet, COCO, ...)."""
        return datumaro.Environment().detect_dataset(path=path)

    def _generate_empty_label_entity(self) -> LabelGroup:
        """Generate Empty Label Group for H-label, Multi-label Classification."""
        empty_label = LabelEntity(name="Empty label", is_empty=True, domain=self.domain)
        empty_group = LabelGroup(name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL)
        return empty_group

    def _generate_default_label_schema(self, label_entities: list) -> LabelSchemaEntity:
        """Generate Default Label Schema for Multi-class Classification, Detecion, Etc."""
        label_schema = LabelSchemaEntity()
        main_group = LabelGroup(
            name="labels",
            labels=label_entities,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        label_schema.add_group(main_group)
        return label_schema

    def _prepare_label_information(
        self,
        datumaro_dataset: dict,
    ) -> dict:
        # Get datumaro category information
        if self.is_train_phase:
            label_categories_list = datumaro_dataset[Subset.TRAINING].categories().get(DatumaroAnnotationType.label, None)
        else:
            label_categories_list = datumaro_dataset[Subset.TESTING].categories().get(DatumaroAnnotationType.label, None)
        category_items = label_categories_list.items

        # Get the 'label_groups' information
        if hasattr(label_categories_list, "label_groups"):
            label_groups = label_categories_list.label_groups
        else:
            label_groups = None

        # LabelEntities
        label_entities = [
            LabelEntity(name=class_name.name, domain=self.domain, is_empty=False, id=ID(i))
            for i, class_name in enumerate(category_items)
        ]

        return {"category_items": category_items, "label_groups": label_groups, "label_entities": label_entities}

    def _select_data_type(self, data_candidates: Union[list, str]) -> str:
        """Select specific type among candidates.

        Args:
            data_candidates (list): Type candidates made by Datumaro.Environment().detect_dataset()

        Returns:
            str: Selected data type
        """
        # TODO: more better way for classification
        if "imagenet" in data_candidates:
            data_type = "imagenet"
        else:
            data_type = data_candidates[0]
        return data_type
