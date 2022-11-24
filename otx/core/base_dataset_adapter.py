"""Interface Class for Data."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member

import abc
from abc import abstractmethod

import datumaro
from datumaro.components.dataset import Dataset as DatumaroDataset

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.label_schema import (LabelGroup, LabelGroupType, LabelSchemaEntity)
from otx.utils.logger import get_logger

logger = get_logger()

class BaseDatasetAdapter(metaclass=abc.ABCMeta):

    def __init__(self, task_type:str):
        self.task_type = task_type
        logger.info('[*] Task type: {}'.format(self.task_type))
        self.domain = task_type.domain

    @abstractmethod
    def import_dataset(
        self,
        train_data_roots: str,
        train_ann_files: str = None,
        val_data_roots: str = None,
        val_ann_files: str = None,
        test_data_roots: str = None,
        test_ann_files: str = None,
        unlabeled_data_roots: str = None,
        unlabeled_file_lists: float = None
    ) -> DatumaroDataset:
        """ Import dataset by using Datumaro.import_from() method.
        
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

        pass

    @abstractmethod
    def convert_to_otx_format(self, datumaro_dataset: dict) -> DatasetEntity:
        """Convert DatumaroDataset to the DatasetEntity.
        Args:
            datumaro_dataset (dict): A Dictionary that includes subset dataset(DatasetEntity) for training/validation/test
        Returns:
            DatasetEntity: 
        """
        pass

    @abstractmethod
    def _auto_split(self):
        """ Automatic train/val/test split. """ 
        #TODO: WIP
        pass

    def _detection_dataset_format(self, path: str) -> str:
        """ Detect dataset format (ImageNet, COCO, ...). """
        return datumaro.Environment().detect_dataset(path=path)

    def _generate_empty_label_entity(self) -> LabelGroup:
        """ Generate Empty Label Group for H-label, Multi-label Classification. """
        empty_label = LabelEntity(name="Empty label", is_empty=True, domain=self.domain)
        empty_group = LabelGroup(name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL)
        return empty_group

    def _generate_default_label_schema(self, label_entities: list) -> LabelSchemaEntity:
        """ Generate Default Label Schema for Multi-class Classification, Detecion, Etc. """
        label_schema = LabelSchemaEntity()
        main_group = LabelGroup(
            name="labels",
            labels=label_entities,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        label_schema.add_group(main_group)
        return label_schema

    @abstractmethod
    def _select_data_type(self, data_candidates: list) -> str:
        """Select specific type among candidates.

        Args:
            data_candidates (list): Type candidates made by Datumaro.Environment().detect_dataset()

        Returns:
            str: Selected data type     
        """
        pass