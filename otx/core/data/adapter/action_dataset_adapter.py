"""Action Base / Classification / Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
import os
import os.path as osp
from typing import Any, Dict, List, Optional

from datumaro.components.annotation import AnnotationType
from datumaro.components.annotation import Bbox as DatumaroBbox
from datumaro.components.dataset import Dataset as DatumaroDataset

from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import LabelEntity
from otx.api.entities.metadata import MetadataItemEntity, VideoMetadata
from otx.api.entities.subset import Subset
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class ActionBaseDatasetAdapter(BaseDatasetAdapter):
    """BaseDataset Adpater for Action tasks inherited by BaseDatasetAdapter."""

    def _import_dataset(
        self,
        train_data_roots: Optional[str] = None,
        val_data_roots: Optional[str] = None,
        test_data_roots: Optional[str] = None,
        unlabeled_data_roots: Optional[str] = None,
    ) -> Dict[Subset, DatumaroDataset]:
        """Import multiple videos that have CVAT format annotation.

        Args:
            train_data_roots (Optional[str]): Path for training data
            val_data_roots (Optional[str]): Path for validation data
            test_data_roots (Optional[str]): Path for test data
            unlabeled_data_roots (Optional[str]): Path for unlabeled data

        Returns:
            DatumaroDataset: Datumaro Dataset
        """
        dataset = {}
        if train_data_roots is None and test_data_roots is None:
            raise ValueError("At least 1 data_root is needed to train/test.")

        if train_data_roots:
            dataset[Subset.TRAINING] = self._prepare_cvat_pair_data(train_data_roots)
            if val_data_roots:
                dataset[Subset.VALIDATION] = self._prepare_cvat_pair_data(val_data_roots)
        if test_data_roots:
            dataset[Subset.TESTING] = self._prepare_cvat_pair_data(test_data_roots)

        return dataset

    def _prepare_cvat_pair_data(self, path: str) -> List[DatumaroDataset]:
        """Preparing a list of DatumaroDataset."""
        cvat_data_list = []
        for cvat_data in os.listdir(path):
            cvat_data_path = osp.join(path, cvat_data)
            cvat_data_list.append(DatumaroDataset.import_from(cvat_data_path, "cvat"))
        return cvat_data_list

    # pylint: disable=protected-access, too-many-nested-blocks
    def _prepare_label_information(self, datumaro_dataset: dict) -> Dict[str, Any]:
        """Prepare and reorganize the label information for merging multiple video information.

        Description w/ examples:

        [Making overall categories]
        Suppose that video1 has labels=[0, 1, 2] and video2 has labels=[0, 1, 4],
        then the overall label should include all label informations as [0, 1, 2, 4].

        [Reindexing the each label index of multiple video datasets]
        In this case, if the label for 'video1/frame_000.jpg' is 2, then the index of label is set to 2.
        For the case of video2, if the label for 'video2/frame_000.jpg' is 4, then the index of label is set to 2.
        However, Since overall labels are [0, 1, 2, 4], 'video2/frame_000.jpg' should has the label index as 3.

        """
        outputs = {
            "label_entities": [],
        }  # type: dict

        # Making overall categories
        has_EmptyFrame = False
        category_names: List[str] = []  # to check the duplicate case
        subset = list(datumaro_dataset.keys())[0]
        for cvat_data in datumaro_dataset[subset]:
            categories = cvat_data.categories().get(AnnotationType.label, None)
            if categories is not None:
                indices = categories._indices
                for name in indices:
                    if name == "EmptyFrame":
                        has_EmptyFrame = True
                        continue
                    if name not in category_names:
                        category_names.append(name)
        category_names.sort()

        if has_EmptyFrame:
            category_names.append("EmptyFrame")

        # Reindexing the each label index of multiple video datasets
        for subset_data in datumaro_dataset.values():
            for cvat_data in subset_data:
                for cvat_data_item in cvat_data:
                    categories = cvat_data.categories().get(AnnotationType.label, None)
                    if categories is not None:
                        for annotation in cvat_data_item.annotations:
                            ann_name = self.get_ann_name(categories._indices, annotation.label)
                            if ann_name is not None:
                                annotation.label = self.get_ann_label(category_names, ann_name)

        # Generate label_entity list according to overall categories
        outputs["label_entities"] = [
            LabelEntity(name=name, domain=self.domain, is_empty=False, id=ID(index))
            for index, name in enumerate(category_names)
        ]
        return outputs

    @staticmethod
    def get_ann_name(indices: Dict[str, int], label: int) -> Optional[str]:
        """Get action name from index."""
        for _name, _label in indices.items():
            if _label == label:
                return _name
        return None

    @staticmethod
    def get_ann_label(category_names: List[str], ann_name: str) -> Optional[int]:
        """Get action index from name."""
        for idx, name in enumerate(category_names):
            if name == ann_name:
                return idx
        return None

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
            for datumaro_items in subset_data:
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    video_name = datumaro_item.media.path.split("/")[-3]
                    shapes: List[Annotation] = []
                    for annotation in datumaro_item.annotations:
                        if annotation.type == AnnotationType.label:
                            shapes.append(self._get_label_entity(annotation))

                    metadata_item = MetadataItemEntity(
                        data=VideoMetadata(
                            video_id=video_name,
                            frame_idx=int(datumaro_item.media.path.split("/")[-1].split(".")[0].lstrip("0")),
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
            for datumaro_items in subset_data:
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    video_name = datumaro_item.media.path.split("/")[-3]
                    shapes: List[Annotation] = []
                    is_empty_frame = False
                    for annotation in datumaro_item.annotations:
                        if isinstance(annotation, DatumaroBbox):
                            if self.label_entities[annotation.label].name == "EmptyFrame":
                                is_empty_frame = True
                                shapes.append(self._get_label_entity(annotation))
                            else:
                                shapes.append(self._get_original_bbox_entity(annotation))
                    metadata_item = MetadataItemEntity(
                        data=VideoMetadata(
                            video_id=video_name,
                            frame_idx=int(datumaro_item.media.path.split("/")[-1].split(".")[0].split("_")[-1]),
                            is_empty_frame=is_empty_frame,
                        )
                    )
                    dataset_item = DatasetItemEntity(
                        image, self._get_ann_scene_entity(shapes), subset=subset, metadata=[metadata_item]
                    )
                    dataset_items.append(dataset_item)

        if self.label_entities[-1].name == "EmptyFrame":
            self.label_entities = self.label_entities[:-1]

        return DatasetEntity(items=dataset_items)
