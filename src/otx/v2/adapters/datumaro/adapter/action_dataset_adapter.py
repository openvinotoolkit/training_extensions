"""Action Base / Classification / Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import os
import os.path as osp
from typing import Dict, List, Optional

from datumaro.components.annotation import AnnotationType as DatumAnnotationType
from datumaro.components.annotation import Bbox as DatumBbox
from datumaro.components.dataset import Dataset as DatumDataset

from otx.v2.api.entities.annotation import Annotation
from otx.v2.api.entities.dataset_item import DatasetItemEntity
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.id import ID
from otx.v2.api.entities.label_schema import LabelSchemaEntity
from otx.v2.api.entities.metadata import MetadataItemEntity, VideoMetadata
from otx.v2.api.entities.subset import Subset

from .datumaro_dataset_adapter import DatumaroDatasetAdapter


class ActionBaseDatasetAdapter(DatumaroDatasetAdapter):
    """BaseDataset Adpater for Action tasks inherited by DatumaroDatasetAdapter."""

    VIDEO_FRAME_SEP = "##"
    EMPTY_FRAME_LABEL_NAME = "EmptyFrame"

    def _import_datasets(
        self,
        train_data_roots: str | None = None,
        train_ann_files: str | None = None,
        val_data_roots: str | None = None,
        val_ann_files: str | None = None,
        test_data_roots: str | None = None,
        test_ann_files: str | None = None,
        unlabeled_data_roots: str | None = None,
        unlabeled_file_list: str | None = None,
        encryption_key: str | None = None,
    ) -> Dict[Subset, DatumDataset]:
        """Import multiple videos that have CVAT format annotation.

        Args:
            train_data_roots (str | None): Path for training data
            train_ann_files (str | None): Path for training annotation file
            val_data_roots (str | None): Path for validation data
            val_ann_files (str | None): Path for validation annotation file
            test_data_roots (str | None): Path for test data
            test_ann_files (str | None): Path for test annotation file
            unlabeled_data_roots (str | None): Path for unlabeled data
            unlabeled_file_list (str | None): Path of unlabeled file list
            encryption_key (str | None): Encryption key to load an encrypted dataset
                                        (only required for DatumaroBinary format)

        Returns:
            DatumDataset: Datumaro Dataset
        """
        dataset = {}
        if train_data_roots is None and test_data_roots is None:
            raise ValueError("At least 1 data_root is needed to train/test.")

        # Construct dataset for training, validation, testing
        if train_data_roots is not None:
            dataset[Subset.TRAINING] = self._prepare_cvat_pair_data(train_data_roots)
            if val_data_roots:
                dataset[Subset.VALIDATION] = self._prepare_cvat_pair_data(val_data_roots)
            self.is_train_phase = True
        if test_data_roots is not None:
            dataset[Subset.TESTING] = self._prepare_cvat_pair_data(test_data_roots)
            self.is_train_phase = False

        return dataset

    def _prepare_cvat_pair_data(self, path: str) -> DatumDataset:
        """Preparing a list of DatumaroDataset."""
        cvat_dataset_list = []
        for video_name in os.listdir(path):
            cvat_data_path = osp.join(path, video_name)
            dataset = DatumDataset.import_from(cvat_data_path, "cvat")
            for item in dataset:
                item.id = f"{video_name}{self.VIDEO_FRAME_SEP}{item.id}"
            cvat_dataset_list.append(dataset)

        dataset = DatumDataset.from_extractors(*cvat_dataset_list, merge_policy="union")
        # set source path for storage cache
        dataset._source_path = path

        # make sure empty frame label has the last label index
        categories = [category.name for category in dataset.categories()[DatumAnnotationType.label]]
        categories.sort()
        dst_labels = [
            (float("inf"), category) if category == self.EMPTY_FRAME_LABEL_NAME else (label, category)
            for label, category in enumerate(categories)
        ]
        dst_labels.sort()
        dst_labels = [name for _, name in dst_labels]
        dataset.transform("project_labels", dst_labels=dst_labels)

        return dataset

    def get_otx_dataset(self) -> dict[Subset, DatumDataset]:
        """Get DatasetEntity.

        Args:
            datumaro_dataset (dict): A Dictionary that includes subset dataset(DatasetEntity)

        Returns:
            DatasetEntity:
        """
        for _, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    video_name, frame_idx = datumaro_item.id.split(self.VIDEO_FRAME_SEP)
                    datumaro_item.attributes['video_id'] = video_name
                    datumaro_item.attributes['frame_idx'] = int(frame_idx)
        return self.dataset


class ActionClassificationDatasetAdapter(ActionBaseDatasetAdapter):
    """Action classification adapter inherited by ActionBaseDatasetAdapter and DatumaroDatasetAdapter."""


class ActionDetectionDatasetAdapter(ActionBaseDatasetAdapter):
    """Action Detection adapter inherited by ActionBaseDatasetAdapter and DatumaroDatasetAdapter."""
