"""Anomaly Classification / Detection / Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
import os
import cv2
import numpy as np
import torch

from datumaro.components.dataset import Dataset as DatumaroDataset

from otx.core.base_dataset_adapter import BaseDatasetAdapter
from otx.api.entities.subset import Subset
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.annotation import (Annotation, AnnotationSceneEntity, AnnotationSceneKind, NullAnnotationSceneEntity)
from otx.api.utils.segmentation_utils import create_annotation_from_segmentation_map
from otx.algorithms.detection.utils.mask_to_bbox import mask2bbox
from otx.utils.logger import get_logger

logger = get_logger()

class AnomalyBaseDatasetAdapter(BaseDatasetAdapter):
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
        # Construct dataset for training, validation, unlabeled
        # TODO: currently, only MVTec dataset format can be used
        self.dataset = {}
        logger.info('[*] Importing Datasets...')
        self.dataset[Subset.TRAINING] = DatumaroDataset.import_from(train_data_roots, format='image_dir')
        self.dataset[Subset.VALIDATION] = DatumaroDataset.import_from(val_data_roots, format='image_dir')
        
        self.gt_path = os.path.join('/'.join(val_data_roots.split('/')[:-1]), 'ground_truth')
        return self.dataset
    
    def _prepare_anomaly_label_information(self) -> dict:
        self.normal_label = LabelEntity(id=ID(0), name="Normal", domain=self.domain)
        self.abnormal_label = LabelEntity(
            id=ID(1),
            name="Anomalous",
            domain=self.domain,
            is_anomalous=True,
        )
        label_entities = [self.normal_label, self.abnormal_label]
        return label_entities

class AnomalyClassificationDatasetAdapter(AnomalyBaseDatasetAdapter, BaseDatasetAdapter):
    def convert_to_otx_format(self, datumaro_dataset: dict) -> DatasetEntity:
        """ Conver DatumaroDataset to DatasetEntity for Anomalytasks. """
        label_entities = self._prepare_anomaly_label_information()
        label_schema = self._generate_default_label_schema(label_entities)

        # Prepare
        dataset_items = []
        for subset, subset_data in datumaro_dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)

                    label = self.normal_label if datumaro_item.id.split('/')[0] == "good" else self.abnormal_label
                    shapes = [
                        Annotation(
                            Rectangle.generate_full_box(),
                            labels=[ScoredLabel(label=label, probability=1.0)],
                        )
                    ]
                    # Unlabeled dataset
                    if len(shapes) == 0:
                        annotation_scene = NullAnnotationSceneEntity()
                    else:
                        annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items), label_schema
        
class AnomalyDetectionDatasetAdapter(AnomalyBaseDatasetAdapter, BaseDatasetAdapter):
    def convert_to_otx_format(self, datumaro_dataset: dict) -> DatasetEntity:
        """ Conver DatumaroDataset to DatasetEntity for Anomalytasks. """
        label_entities = self._prepare_anomaly_label_information()
        label_schema = self._generate_default_label_schema(label_entities)

        # Prepare
        dataset_items = []
        for subset, subset_data in datumaro_dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    label = self.normal_label if datumaro_item.id.split('/')[0] == "good" else self.abnormal_label
                    shapes = [
                        Annotation(
                            Rectangle.generate_full_box(),
                            labels=[ScoredLabel(label=label, probability=1.0)],
                        )
                    ]
                    mask_file_path = os.path.join(self.gt_path, str(datumaro_item.id) +'_mask.png')
                    if os.path.exists(mask_file_path):
                        mask = (cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)
                        bboxes = mask2bbox(mask)
                        for bbox in bboxes:
                            x1, y1, x2, y2 = bbox
                            shapes.append(
                                Annotation(
                                    Rectangle(
                                        x1=x1/image.width,
                                        y1=y1/image.height,
                                        x2=x2/image.width,
                                        y2=y2/image.height,
                                    ),
                                    labels = [
                                        ScoredLabel(
                                            label=self.abnormal_label
                                        )
                                    ]
                                )
                            )
                    # Unlabeled dataset
                    if len(shapes) == 0:
                        annotation_scene = NullAnnotationSceneEntity()
                    else:
                        annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items), label_schema

class AnomalySegmentationDatasetAdapter(AnomalyBaseDatasetAdapter, BaseDatasetAdapter):
    def convert_to_otx_format(self, datumaro_dataset: dict) -> DatasetEntity:
        """ Conver DatumaroDataset to DatasetEntity for Anomalytasks. """
        label_entities = self._prepare_anomaly_label_information()
        label_schema = self._generate_default_label_schema(label_entities)

        # Prepare
        dataset_items = []
        for subset, subset_data in datumaro_dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    label = self.normal_label if datumaro_item.id.split('/')[0] == "good" else self.abnormal_label
                    shapes = [
                        Annotation(
                            Rectangle.generate_full_box(),
                            labels=[ScoredLabel(label=label, probability=1.0)],
                        )
                    ]
                    mask_file_path = os.path.join(self.gt_path, str(datumaro_item.id) +'_mask.png')
                    if os.path.exists(mask_file_path):
                        mask = (cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)
                        shapes.extend(
                            create_annotation_from_segmentation_map(
                                hard_prediction = mask, 
                                soft_prediction = np.ones_like(mask),
                                label_map = {
                                    0: self.normal_label,
                                    1: self.abnormal_label
                                }
                            )
                        )
                    # Unlabeled dataset
                    if len(shapes) == 0:
                        annotation_scene = NullAnnotationSceneEntity()
                    else:
                        annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items), label_schema
