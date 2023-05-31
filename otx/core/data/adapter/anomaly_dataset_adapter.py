"""Anomaly Classification / Detection / Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member, too-many-arguments
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from datumaro.components.dataset import Dataset as DatumaroDataset

from otx.algorithms.common.utils.mask_to_bbox import mask2bbox
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
    NullAnnotationSceneEntity,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.utils.segmentation_utils import create_annotation_from_segmentation_map
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class AnomalyBaseDatasetAdapter(BaseDatasetAdapter):
    """BaseDataset Adpater for Anomaly tasks inherited from BaseDatasetAdapter."""

    def _import_datasets(
        self,
        train_data_roots: Optional[str] = None,
        train_ann_files: Optional[str] = None,
        val_data_roots: Optional[str] = None,
        val_ann_files: Optional[str] = None,
        test_data_roots: Optional[str] = None,
        test_ann_files: Optional[str] = None,
        unlabeled_data_roots: Optional[str] = None,
        unlabeled_file_list: Optional[str] = None,
        encryption_key: Optional[str] = None,
    ) -> Dict[Subset, DatumaroDataset]:
        """Import MVTec dataset.

        Args:
            train_data_roots (Optional[str]): Path for training data
            train_ann_files (Optional[str]): Path for training annotation file
            val_data_roots (Optional[str]): Path for validation data
            val_ann_files (Optional[str]): Path for validation annotation file
            test_data_roots (Optional[str]): Path for test data
            test_ann_files (Optional[str]): Path for test annotation file
            unlabeled_data_roots (Optional[str]): Path for unlabeled data
            unlabeled_file_list (Optional[str]): Path of unlabeled file list
            encryption_key (Optional[str]): Encryption key to load an encrypted dataset
                                        (only required for DatumaroBinary format)

        Returns:
            DatumaroDataset: Datumaro Dataset
        """
        # Construct dataset for training, validation, unlabeled
        # TODO: currently, only MVTec dataset format can be used
        dataset = {}
        if train_data_roots is None and test_data_roots is None:
            raise ValueError("At least 1 data_root is needed to train/test.")

        if train_data_roots:
            dataset[Subset.TRAINING] = DatumaroDataset.import_from(train_data_roots, format="image_dir")
            if val_data_roots:
                dataset[Subset.VALIDATION] = DatumaroDataset.import_from(val_data_roots, format="image_dir")
            else:
                raise NotImplementedError("Anomaly task needs validation dataset.")
        if test_data_roots:
            dataset[Subset.TESTING] = DatumaroDataset.import_from(test_data_roots, format="image_dir")
        return dataset

    def _prepare_anomaly_label_information(self) -> List[LabelEntity]:
        """Prepare LabelEntity List."""
        normal_label = LabelEntity(id=ID(0), name="Normal", domain=self.domain)
        abnormal_label = LabelEntity(
            id=ID(1),
            name="Anomalous",
            domain=self.domain,
            is_anomalous=True,
        )
        return [normal_label, abnormal_label]

    def get_otx_dataset(self) -> DatasetEntity:
        """Get DatasetEntity."""
        raise NotImplementedError()


class AnomalyClassificationDatasetAdapter(AnomalyBaseDatasetAdapter):
    """Anomaly classification adapter inherited from AnomalyBaseDatasetAdapter."""

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Anomaly classification."""
        normal_label, abnormal_label = self._prepare_anomaly_label_information()
        self.label_entities = [normal_label, abnormal_label]

        # Prepare
        dataset_items: List[DatasetItemEntity] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    label = normal_label if os.path.dirname(datumaro_item.id) == "good" else abnormal_label
                    shapes = [
                        Annotation(
                            Rectangle.generate_full_box(),
                            labels=[ScoredLabel(label=label, probability=1.0)],
                        )
                    ]
                    annotation_scene: Optional[AnnotationSceneEntity] = None
                    # Unlabeled dataset
                    if len(shapes) == 0:
                        annotation_scene = NullAnnotationSceneEntity()
                    else:
                        annotation_scene = AnnotationSceneEntity(
                            kind=AnnotationSceneKind.ANNOTATION, annotations=shapes
                        )
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items)


class AnomalyDetectionDatasetAdapter(AnomalyBaseDatasetAdapter):
    """Anomaly detection adapter inherited from AnomalyBaseDatasetAdapter."""

    def get_otx_dataset(self) -> DatasetEntity:
        """Conver DatumaroDataset to DatasetEntity for Anomaly detection."""
        normal_label, abnormal_label = self._prepare_anomaly_label_information()
        self.label_entities = [normal_label, abnormal_label]

        # Prepare
        dataset_items: List[DatasetItemEntity] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    label = normal_label if os.path.dirname(datumaro_item.id) == "good" else abnormal_label
                    shapes = [
                        Annotation(
                            Rectangle.generate_full_box(),
                            labels=[ScoredLabel(label=label, probability=1.0)],
                        )
                    ]
                    # TODO: avoid hard coding, plan to enable MVTec to Datumaro
                    mask_file_path = os.path.join(
                        "/".join(datumaro_item.media.path.split("/")[:-3]),
                        "ground_truth",
                        str(datumaro_item.id) + "_mask.png",
                    )
                    if os.path.exists(mask_file_path):
                        mask = (cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE) / 255).astype(np.uint8)
                        bboxes = mask2bbox(mask)
                        for bbox in bboxes:
                            x1, y1, x2, y2 = bbox
                            shapes.append(
                                Annotation(
                                    Rectangle(
                                        x1=x1 / image.width,
                                        y1=y1 / image.height,
                                        x2=x2 / image.width,
                                        y2=y2 / image.height,
                                    ),
                                    labels=[ScoredLabel(label=abnormal_label)],
                                )
                            )
                    annotation_scene: Optional[AnnotationSceneEntity] = None
                    # Unlabeled dataset
                    if len(shapes) == 0:
                        annotation_scene = NullAnnotationSceneEntity()
                    else:
                        annotation_scene = AnnotationSceneEntity(
                            kind=AnnotationSceneKind.ANNOTATION, annotations=shapes
                        )
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items)


class AnomalySegmentationDatasetAdapter(AnomalyBaseDatasetAdapter):
    """Anomaly segmentation adapter inherited by AnomalyBaseDatasetAdapter and BaseDatasetAdapter."""

    def get_otx_dataset(self) -> DatasetEntity:
        """Conver DatumaroDataset to DatasetEntity for Anomaly segmentation."""
        normal_label, abnormal_label = self._prepare_anomaly_label_information()
        self.label_entities = [normal_label, abnormal_label]

        # Prepare
        dataset_items: List[DatasetItemEntity] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    label = normal_label if os.path.dirname(datumaro_item.id) == "good" else abnormal_label
                    shapes = [
                        Annotation(
                            Rectangle.generate_full_box(),
                            labels=[ScoredLabel(label=label, probability=1.0)],
                        )
                    ]
                    # TODO: avoid hard coding, plan to enable MVTec to Datumaro
                    mask_file_path = os.path.join(
                        "/".join(datumaro_item.media.path.split("/")[:-3]),
                        "ground_truth",
                        str(datumaro_item.id) + "_mask.png",
                    )
                    if os.path.exists(mask_file_path):
                        mask = (cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE) / 255).astype(np.uint8)
                        shapes.extend(
                            create_annotation_from_segmentation_map(
                                hard_prediction=mask,
                                soft_prediction=np.ones_like(mask),
                                label_map={0: normal_label, 1: abnormal_label},
                            )
                        )
                    annotation_scene: Optional[AnnotationSceneEntity] = None
                    # Unlabeled dataset
                    if len(shapes) == 0:
                        annotation_scene = NullAnnotationSceneEntity()
                    else:
                        annotation_scene = AnnotationSceneEntity(
                            kind=AnnotationSceneKind.ANNOTATION, annotations=shapes
                        )
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items)
