"""Action Base / Classification / Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member, too-many-arguments
import os
import os.path as osp
from typing import Dict, List, Optional

from datumaro.components.annotation import AnnotationType
from datumaro.components.annotation import Bbox as DatumBbox
from datumaro.components.dataset import Dataset as DatumDataset

from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.metadata import MetadataItemEntity, VideoMetadata
from otx.api.entities.subset import Subset
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class ActionBaseDatasetAdapter(BaseDatasetAdapter):
    """BaseDataset Adpater for Action tasks inherited by BaseDatasetAdapter."""

    VIDEO_FRAME_SEP = "##"
    EMPTY_FRAME_LABEL_NAME = "EmptyFrame"

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
    ) -> Dict[Subset, DatumDataset]:
        """Import multiple videos that have CVAT format annotation.

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
        if test_data_roots is not None and train_data_roots is None:
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
        categories = [category.name for category in dataset.categories()[AnnotationType.label]]
        categories.sort()
        dst_labels = [
            (float("inf"), category) if category == self.EMPTY_FRAME_LABEL_NAME else (label, category)
            for label, category in enumerate(categories)
        ]
        dst_labels.sort()
        dst_labels = [name for _, name in dst_labels]
        dataset.transform("project_labels", dst_labels=dst_labels)

        return dataset

    def get_otx_dataset(self) -> DatasetEntity:
        """Get DatasetEntity.

        Args:
            datumaro_dataset (dict): A Dictionary that includes subset dataset(DatasetEntity)

        Returns:
            DatasetEntity:
        """
        raise NotImplementedError()


class ActionClassificationDatasetAdapter(ActionBaseDatasetAdapter):
    """Action classification adapter inherited by ActionBaseDatasetAdapter and BaseDatasetAdapter."""

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Acion Classification."""
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        dataset_items: List[DatasetItemEntity] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = self.datum_media_2_otx_media(datumaro_item.media)
                    assert isinstance(image, Image)
                    shapes: List[Annotation] = []
                    for annotation in datumaro_item.annotations:
                        if annotation.type == AnnotationType.label:
                            shapes.append(self._get_label_entity(annotation))

                    video_name, frame_idx = datumaro_item.id.split(self.VIDEO_FRAME_SEP)
                    metadata_item = MetadataItemEntity(
                        data=VideoMetadata(
                            video_id=video_name,
                            frame_idx=int(frame_idx),
                            is_empty_frame=False,
                        )
                    )

                    dataset_item = DatasetItemEntity(
                        image, self._get_ann_scene_entity(shapes), subset=subset, metadata=[metadata_item]
                    )
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items)


class ActionDetectionDatasetAdapter(ActionBaseDatasetAdapter):
    """Action Detection adapter inherited by ActionBaseDatasetAdapter and BaseDatasetAdapter."""

    # pylint: disable=too-many-nested-blocks
    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Acion Detection."""
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        # Detection use index 0 as a background category
        for label_entity in self.label_entities:
            label_entity.id = ID(int(label_entity.id) + 1)

        dataset_items: List[DatasetItemEntity] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = self.datum_media_2_otx_media(datumaro_item.media)
                    assert isinstance(image, Image)
                    shapes: List[Annotation] = []
                    is_empty_frame = False
                    for annotation in datumaro_item.annotations:
                        if isinstance(annotation, DatumBbox):
                            if self.label_entities[annotation.label].name == self.EMPTY_FRAME_LABEL_NAME:
                                is_empty_frame = True
                                shapes.append(self._get_label_entity(annotation))
                            else:
                                shapes.append(self._get_original_bbox_entity(annotation))

                    video_name, frame_name = datumaro_item.id.split(self.VIDEO_FRAME_SEP)
                    metadata_item = MetadataItemEntity(
                        data=VideoMetadata(
                            video_id=video_name,
                            frame_idx=int(frame_name.split("_")[-1]),
                            is_empty_frame=is_empty_frame,
                        )
                    )
                    dataset_item = DatasetItemEntity(
                        image, self._get_ann_scene_entity(shapes), subset=subset, metadata=[metadata_item]
                    )
                    dataset_items.append(dataset_item)

        found = [i for i, entity in enumerate(self.label_entities) if entity.name == self.EMPTY_FRAME_LABEL_NAME]
        if found:
            self.label_entities.pop(found[0])

        return DatasetEntity(items=dataset_items)
