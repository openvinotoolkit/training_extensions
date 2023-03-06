"""Anomaly Classification / Detection / Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
import os
from typing import List, Optional, Union

import cv2
import numpy as np

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
from otx.api.entities.model_template import TaskType
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.utils.segmentation_utils import create_annotation_from_segmentation_map
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class AnomalyBaseDatasetAdapter(BaseDatasetAdapter):
    """BaseDataset Adpater for Anomaly tasks inherited from BaseDatasetAdapter."""

    def _select_data_type(self, data_candidates: Union[List[str], str]) -> str:
        """Select specific type among candidates.

        Args:
            data_candidates (Union[List[str], str]): Type candidates made by Datumaro.Environment().detect_dataset()

        Returns:
            str: Selected data type
        """
        if self.task_type == TaskType.ACTION_CLASSIFICATION:
            assert "mvtec_classification" in data_candidates
            data_type = "mvtec_classification"
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            assert "mvtec_segmentation" in data_candidates
            data_type = "mvtec_segmentation"
        elif self.task_type == TaskType.ANOMALY_DETECTION:
            assert "mvtec_detection" in data_candidates
            data_type = "mvtec_detection"
        else:
            raise ValueError(f"Unrecognized anomaly task type: {self.task_type}")
        return data_type

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
