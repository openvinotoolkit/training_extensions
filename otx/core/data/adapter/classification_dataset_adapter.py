"""Classification Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
from typing import List, Union

from datumaro.components.annotation import AnnotationType, LabelCategories

from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class ClassificationDatasetAdapter(BaseDatasetAdapter):
    """Classification adapter inherited from BaseDatasetAdapter.

    It converts DatumaroDataset -> DatasetEntity
    for multi-class, multi-label, and hierarchical-label classification tasks
    """

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Classification."""
        # Prepare label information
        label_information = self._prepare_label_information(self.dataset)
        self.category_items = label_information["category_items"]
        self.label_groups = label_information["label_groups"]
        self.label_entities = label_information["label_entities"]

        # Generate label schema
        self.label_schema = self._generate_classification_label_schema(self.label_groups, self.label_entities)

        # Set the DatasetItemEntity
        dataset_items: List[DatasetItemEntity] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    datumaro_labels = []
                    for ann in datumaro_item.annotations:
                        if ann.type == AnnotationType.label:
                            datumaro_labels.append(ann.label)

                    shapes = self._get_cls_shapes(datumaro_labels)
                    dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)

                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items)

    def _get_cls_shapes(self, datumaro_labels: List[int]) -> List[Annotation]:
        """Converts a list of datumaro labels to Annotation object."""
        otx_labels = []
        for d_label in datumaro_labels:
            otx_labels.append(ScoredLabel(label=self.label_entities[d_label], probability=1.0))

        return [Annotation(Rectangle.generate_full_box(), labels=otx_labels)]

    def get_label_schema(self) -> LabelSchemaEntity:
        """Get Label Schema."""
        return self._generate_classification_label_schema(self.label_groups, self.label_entities)

    def _generate_classification_label_schema(
        self, label_groups: List[LabelCategories.LabelGroup], label_entities: List[LabelEntity]
    ) -> LabelSchemaEntity:
        """Generate LabelSchema for Classification."""
        label_schema = LabelSchemaEntity()

        if len(label_groups) > 0:
            for label_group in label_groups:
                group_label_entity_list = []
                for label in label_group.labels:
                    label_entity = [le for le in label_entities if le.name == label]
                    group_label_entity_list.append(label_entity[0])

                label_schema.add_group(
                    LabelGroup(
                        name=label_group.name, labels=group_label_entity_list, group_type=LabelGroupType.EXCLUSIVE
                    )
                )
            label_schema.add_group(self._generate_empty_label_entity())
        else:
            label_schema = self._generate_default_label_schema(label_entities)

        return label_schema

    def _select_data_type(self, data_candidates: Union[list, str]) -> str:
        return "imagenet" if "imagenet" in data_candidates else data_candidates[0]
