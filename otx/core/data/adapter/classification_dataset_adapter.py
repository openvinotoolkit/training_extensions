"""Classification Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
from typing import List, Union

from datumaro.components.annotation import AnnotationType as DatumAnnotationType
from datumaro.components.annotation import LabelCategories as DatumLabelCategories

from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntityWithID
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class ClassificationDatasetAdapter(BaseDatasetAdapter):
    """Classification adapter inherited from BaseDatasetAdapter.

    It converts DatumaroDataset -> DatasetEntity
    for multi-class, multi-label, and hierarchical-label classification tasks
    """

    def _get_dataset_items(self, fake_ann=False):
        # Set the DatasetItemEntityWithID
        dataset_items: List[DatasetItemEntityWithID] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = self.datum_media_2_otx_media(datumaro_item.media)
                    assert isinstance(image, Image)
                    if not fake_ann:
                        datumaro_labels = []
                        for ann in datumaro_item.annotations:
                            if ann.type == DatumAnnotationType.label:
                                datumaro_labels.append(ann.label)
                    else:
                        datumaro_labels = [0]  # fake label

                    shapes = self._get_cls_shapes(datumaro_labels)
                    dataset_item = DatasetItemEntityWithID(
                        image, self._get_ann_scene_entity(shapes), subset=subset, id_=datumaro_item.id
                    )

                    dataset_items.append(dataset_item)
        return dataset_items

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Classification."""
        # Prepare label information
        label_information = self._prepare_label_information(self.dataset)
        self.category_items = label_information["category_items"]
        self.label_groups = label_information["label_groups"]
        self.label_entities = label_information["label_entities"]
        dataset_items = self._get_dataset_items()
        return DatasetEntity(items=dataset_items)

    def _get_cls_shapes(self, datumaro_labels: List[int]) -> List[Annotation]:
        """Converts a list of datumaro labels to Annotation object."""
        otx_labels = []
        for d_label in datumaro_labels:
            otx_labels.append(ScoredLabel(label=self.label_entities[d_label], probability=1.0))

        return [Annotation(Rectangle.generate_full_box(), labels=otx_labels)]

    def get_label_schema(self) -> LabelSchemaEntity:
        """Get Label Schema."""
        return self._generate_classification_label_schema(
            self.category_items,
            self.label_groups,
            self.label_entities,
        )

    def _generate_classification_label_schema(
        self,
        category_items: List[DatumLabelCategories.Category],
        label_groups: List[DatumLabelCategories.LabelGroup],
        label_entities: List[LabelEntity],
    ) -> LabelSchemaEntity:
        """Generate LabelSchema for Classification."""
        label_schema = LabelSchemaEntity()

        # construct label group
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

        # construct label tree
        for category_item in category_items:
            me = [i for i in label_entities if i.name == category_item.name]
            parent = [i for i in label_entities if i.name == category_item.parent]
            if len(me) != 1:
                raise ValueError(
                    f"Label name must be unique but {len(me)} labels found for label name '{category_item.name}'."
                )
            if len(parent) == 0:
                label_schema.label_tree.add_node(me[0])
            elif len(parent) == 1:
                label_schema.add_child(parent[0], me[0])
            else:
                raise ValueError(
                    f"Label name must be unique but {len(parent)} labels found for label name '{category_item.parent}'."
                )

        return label_schema

    def _select_data_type(self, data_candidates: Union[list, str]) -> str:
        return "imagenet" if "imagenet" in data_candidates else data_candidates[0]


class SelfSLClassificationDatasetAdapter(ClassificationDatasetAdapter):
    """SelfSLClassification adapter inherited from ClassificationDatasetAdapter.

    It creates fake annotations to work with DatumaroDataset w/o labels
    and converts it to DatasetEntity for Self-SL classification pretraining
    """

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Self-SL Classification."""
        # Prepare label information
        if not self.dataset[Subset.TRAINING].categories():
            label_information = self._prepare_fake_label_information()
            self.category_items = label_information["category_items"]
            self.label_groups = label_information["label_groups"]
            self.label_entities = label_information["label_entities"]
            dataset_items = self._get_dataset_items(fake_ann=True)
            return DatasetEntity(items=dataset_items)
        return super().get_otx_dataset()

    def _prepare_fake_label_information(self):
        label_categories_list = DatumLabelCategories.from_iterable(["fake_label"])
        category_items = label_categories_list.items
        label_groups = label_categories_list.label_groups
        # LabelEntities
        label_entities = [LabelEntity(name="fake_label", domain=self.domain, is_empty=False, id=ID(0))]
        return {"category_items": category_items, "label_groups": label_groups, "label_entities": label_entities}
