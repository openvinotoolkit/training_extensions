"""Anomaly Classification / Detection / Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from datumaro.components.annotation import Bbox as DatumBbox
from datumaro.components.annotation import Label as DatumLabel
from datumaro.components.annotation import Mask as DatumMask
from datumaro.components.dataset import Dataset as DatumDataset
from datumaro.components.annotation import Categories as DatumCategories

from otx.v2.adapters.torch.modules.utils.mask_to_bbox import mask2bbox
from otx.v2.api.entities.id import ID
from otx.v2.api.entities.label import LabelEntity
from otx.v2.api.entities.label_schema import LabelSchemaEntity
from otx.v2.api.entities.subset import Subset

from .datumaro_dataset_adapter import DatumaroDatasetAdapter

LabelInformationType = Dict[str, Union[List[LabelEntity], List[DatumCategories]]]

class AnomalyBaseDatasetAdapter(DatumaroDatasetAdapter):
    """BaseDataset Adpater for Anomaly tasks inherited from DatumaroDatasetAdapter."""

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
    ) -> dict[Subset, DatumDataset]:
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
        dataset = {}
        if train_data_roots is None and test_data_roots is None:
            raise ValueError("At least 1 data_root is needed to train/test.")

        if train_data_roots:
            dataset[Subset.TRAINING] = DatumDataset.import_from(train_data_roots, format="image_dir")
            if val_data_roots:
                dataset[Subset.VALIDATION] = DatumDataset.import_from(val_data_roots, format="image_dir")
        if test_data_roots:
            dataset[Subset.TESTING] = DatumDataset.import_from(test_data_roots, format="image_dir")
        return dataset

    def _prepare_label_information(self, datumaro_dataset: dict[Subset, DatumDataset]) -> LabelInformationType:
        """Prepare LabelEntity List."""
        normal_label = LabelEntity(id=ID(0), name="Normal", domain=self.domain)
        abnormal_label = LabelEntity(
            id=ID(1),
            name="Anomalous",
            domain=self.domain,
            is_anomalous=True,
        )
        return {
            "category_items": [],
            "label_groups": [],
            "label_entities": [normal_label, abnormal_label]
        }

    def get_label_schema(self) -> LabelSchemaEntity:
        """Get Label Schema."""
        return super()._generate_default_label_schema(self.label_entities)

    def get_otx_dataset(self) -> dict[Subset, DatumDataset]:
        """Get DatasetEntity."""
        raise NotImplementedError


class AnomalyClassificationDatasetAdapter(AnomalyBaseDatasetAdapter):
    """Anomaly classification adapter inherited from AnomalyBaseDatasetAdapter."""

    def get_otx_dataset(self) -> dict[Subset, DatumDataset]:
        """Return DatumaroDataset for Anomaly classification."""
        # Prepare
        for _, subset_data in self.dataset.items():
            for item in subset_data:
                label = self.label_entities[0] if os.path.dirname(item.id) == "good" else self.label_entities[1]
                item.annotations = [DatumLabel(label=label.id, attributes={"is_anomalous": label.is_anomalous})]

        return self.dataset


class AnomalyDetectionDatasetAdapter(AnomalyBaseDatasetAdapter):
    """Anomaly detection adapter inherited from AnomalyBaseDatasetAdapter."""

    def get_otx_dataset(self) -> dict[Subset, DatumDataset]:
        """Convert Mask annotation into Polygon in DatumaroDataset."""

        # Prepare
        for _, subset_data in self.dataset.items():
            for item in subset_data:
                image = item.media.data
                label = self.label_entities[0] if os.path.dirname(item.id) == "good" else self.label_entities[1]
                annotations = [] # [DatumLabel(label.id)]

                mask_file_path = os.path.join(
                    "/".join(item.media.path.split("/")[:-3]),
                    "ground_truth",
                    str(item.id) + "_mask.png",
                )
                if os.path.exists(mask_file_path):
                    mask = (cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE) / 255).astype(np.uint8)
                    bboxes = mask2bbox(mask)
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        annotations.append(
                            DatumBbox(
                                x=x1 / image.width,
                                y=y1 / image.height,
                                w=(x2 - x1) / image.width,
                                h=(y2 - y1) / image.height,
                                label=self.label_entities[1].id,
                                attributes={"is_anomalous": label.is_anomalous},
                            ),
                        )
                item.annotations = annotations

        return self.dataset


class AnomalySegmentationDatasetAdapter(AnomalyBaseDatasetAdapter):
    """Anomaly segmentation adapter inherited by AnomalyBaseDatasetAdapter and DatumaroDatasetAdapter."""

    def get_otx_dataset(self) -> dict[Subset, DatumDataset]:
        """Conver DatumaroDataset to DatasetEntity for Anomaly segmentation."""

        # Prepare
        for _, subset_data in self.dataset.items():
            for item in subset_data:
                label = self.label_entities[0] if os.path.dirname(item.id) == "good" else self.label_entities[1]
                annotations = [] #[DatumLabel(label.id)]

                mask_file_path = os.path.join(
                    "/".join(item.media.path.split("/")[:-3]),
                    "ground_truth",
                    str(item.id) + "_mask.png",
                )
                if os.path.exists(mask_file_path):
                    mask = (cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE) / 255).astype(np.uint8)
                    annotations.append(
                        DatumMask(
                            image=mask,
                            label=self.label_entities[1].id,
                            attributes={
                                "label_map": {0: self.label_entities[0], 1: self.label_entities[1]},
                                "is_anomalous": label.is_anomalous,
                            },
                        ),
                    )
                item.annotations = annotations

        return self.dataset
