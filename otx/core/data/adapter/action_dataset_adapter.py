"""Action Base / Classification / Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
import os
import os.path as osp
from typing import Dict, List

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import Dataset as DatumaroDataset

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
        train_data_roots: str = None,
        val_data_roots: str = None,
        test_data_roots: str = None,
        unlabeled_data_roots: str = None,
    ) -> Dict[Subset, DatumaroDataset]:
        """Import multiple videos that have CVAT format annotation."""
        dataset = {}
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

    def _prepare_label_information(self, datumaro_dataset: dict) -> dict:
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
            "category_items": [],
            "label_groups": [],
            "label_entities": [],
        }  # type: dict

        category_list = []  # to check the duplicate case
        for cvat_data in datumaro_dataset[Subset.TRAINING]:
            # Making overall categories
            categories = cvat_data.categories().get(AnnotationType.label, None)

            if categories not in category_list:
                outputs["category_items"].extend(categories.items)
                outputs["label_groups"].extend(categories.label_groups)

            category_list.append(categories)

            # Reindexing the each label index of multiple video datasets
            for cvat_data_item in cvat_data:
                for ann in cvat_data_item.annotations:
                    ann_name = categories.items[ann.label].name
                    ann.label = [
                        i for i, category_item in enumerate(outputs["category_items"]) if category_item.name == ann_name
                    ][0]

        # Generate label_entity list according to overall categories
        outputs["label_entities"] = [
            LabelEntity(name=class_name.name, domain=self.domain, is_empty=False, id=ID(i))
            for i, class_name in enumerate(outputs["category_items"])
        ]
        return outputs

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

        dataset_items = []
        for subset, subset_data in self.dataset.items():
            for datumaro_items in subset_data:
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    shapes = []
                    for ann in datumaro_item.annotations:
                        if ann.type == AnnotationType.label:
                            shapes.append(self._get_label_entity(ann))

                    meta_item = [
                        MetadataItemEntity(
                            data=VideoMetadata(
                                name="video_meta",
                                video_id=int(datumaro_item.media.path.split("/")[-3].split("_")[-1]),
                                frame_idx=int(datumaro_item.media.path.split("/")[-1].split(".")[0].lstrip("0")),
                            )
                        )
                    ]

                    dataset_item = DatasetItemEntity(
                        image, self._get_ann_scene_entity(shapes), subset=subset, metadata=meta_item
                    )
                    dataset_items.append(dataset_item)
        return DatasetEntity(items=dataset_items)


class ActionDetectionDatasetAdapter(ActionBaseDatasetAdapter):
    """Action Detection adapter inherited by ActionBaseDatasetAdapter and BaseDatasetAdapter."""

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Acion Detection."""
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        dataset_items = []
        for subset, subset_data in self.dataset.items():
            for datumaro_items in subset_data:
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    shapes = []
                    for ann in datumaro_item.annotations:
                        if ann.type == AnnotationType.bbox:
                            shapes.append(self._get_original_bbox_entity(ann))

                    dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items)
