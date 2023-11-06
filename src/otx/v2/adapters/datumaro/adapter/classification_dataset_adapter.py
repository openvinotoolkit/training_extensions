"""Classification Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, List, Union

from datumaro.components.annotation import LabelCategories as DatumLabelCategories
from datumaro.components.dataset import Dataset as DatumDataset

from otx.v2.api.entities.annotation import Annotation
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.id import ID
from otx.v2.api.entities.label import LabelEntity
from otx.v2.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.v2.api.entities.scored_label import ScoredLabel
from otx.v2.api.entities.shapes.rectangle import Rectangle
from otx.v2.api.entities.subset import Subset

from .datumaro_dataset_adapter import DatumaroDatasetAdapter


class ClassificationDatasetAdapter(DatumaroDatasetAdapter):
    """Classification adapter inherited from DatumaroDatasetAdapter.

    It converts DatumaroDataset -> DatasetEntity
    for multi-class, multi-label, and hierarchical-label classification tasks
    """

    def get_otx_dataset(self) -> Dict[Subset, DatumDataset]:
        """Convert DatumaroDataset to DatasetEntity for Classification."""
        return self.dataset

    def _get_cls_shapes(self, datumaro_labels: List[int]) -> List[Annotation]:
        """Converts a list of datumaro labels to Annotation object."""
        otx_labels = []
        for d_label in datumaro_labels:
            otx_labels.append(ScoredLabel(label=self.label_entities[d_label], probability=1.0))

        return [Annotation(Rectangle.generate_full_box(), labels=otx_labels)]

    def get_label_schema(self) -> LabelSchemaEntity:
        """Get Label Schema."""
        label_information = self._prepare_label_information(self.dataset)
        self.category_items = label_information["category_items"]
        self.label_groups = label_information["label_groups"]
        self.label_entities = label_information["label_entities"]

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
                        name=label_group.name,
                        labels=group_label_entity_list,
                        group_type=LabelGroupType.EXCLUSIVE,
                    ),
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
                    f"Label name must be unique but {len(me)} labels found for label name '{category_item.name}'.",
                )
            if len(parent) == 0:
                label_schema.label_tree.add_node(me[0])
            elif len(parent) == 1:
                label_schema.add_child(parent[0], me[0])
            else:
                raise ValueError(
                    f"Label name must be unique but {len(parent)} labels found for label name '{category_item.parent}'.",
                )

        return label_schema

    def _select_data_type(self, data_candidates: Union[list, str]) -> str:
        return "imagenet" if "imagenet" in data_candidates else data_candidates[0]


class SelfSLClassificationDatasetAdapter(ClassificationDatasetAdapter):
    """SelfSLClassification adapter inherited from ClassificationDatasetAdapter.

    It creates fake annotations to work with DatumaroDataset w/o labels
    and converts it to DatasetEntity for Self-SL classification pretraining
    """

    def get_label_schema(self) -> LabelSchemaEntity:
        """Get Label Schema."""
        if not self.dataset[Subset.TRAINING].categories():
            label_information = self._prepare_fake_label_information()
        else:
            label_information = self._prepare_label_information(self.dataset)
        self.category_items = label_information["category_items"]
        self.label_groups = label_information["label_groups"]
        self.label_entities = label_information["label_entities"]
        return self._generate_default_label_schema(self.label_entities)

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Self-SL Classification."""
        self._get_dataset_items(fake_ann=True)
        return self.dataset

    def _prepare_fake_label_information(self) -> Dict[str, Any]:
        label_categories_list = DatumLabelCategories.from_iterable(["fake_label"])
        category_items = label_categories_list.items
        label_groups = label_categories_list.label_groups
        # LabelEntities
        label_entities = [LabelEntity(name="fake_label", domain=self.domain, is_empty=False, id=ID(0))]
        return {"category_items": category_items, "label_groups": label_groups, "label_entities": label_entities}

    def _get_dataset_items(self, fake_ann: bool = False):       
         # Set the DatasetItemEntityWithID
        for _, subset in self.dataset.items():
            for item in subset:
                if fake_ann:
                    item.annotations = [0]
