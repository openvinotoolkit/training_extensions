"""Classification Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
from typing import List
from datumaro.components.annotation import LabelCategories

from otx.core.base_dataset_adapter import BaseDatasetAdapter
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.subset import Subset
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import (LabelGroup, LabelGroupType, LabelSchemaEntity)
from otx.api.entities.annotation import (Annotation, AnnotationSceneEntity, AnnotationSceneKind, NullAnnotationSceneEntity)
from otx.utils.logger import get_logger

logger = get_logger()

class ClassificationDatasetAdapter(BaseDatasetAdapter):
    def convert_to_otx_format(self, datumaro_dataset: dict) -> DatasetEntity:
        """ Convert DatumaroDataset to DatasetEntity for Classification. """
        # Prepare label information
        label_information = self._prepare_label_information(datumaro_dataset)
        category_items = label_information["category_items"]
        label_groups = label_information["label_groups"]
        label_entities = label_information["label_entities"]
        
        # Generate label schema
        label_schema = self._generate_classification_label_schema(label_groups, label_entities)

        # Set the DatasetItemEntity
        dataset_items = []
        for subset, subset_data in datumaro_dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    labels = [
                        ScoredLabel(
                            label= [label for label in label_entities if label.name == category_items[ann.label].name][0],
                            probability=1.0   
                        ) for ann in datumaro_item.annotations
                    ]
                    shapes = [Annotation(Rectangle.generate_full_box(), labels)]
                    
                    # Unlabeled dataset
                    if len(shapes) == 0:
                        annotation_scene = NullAnnotationSceneEntity()
                    else:
                        annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items), label_schema

    def _generate_classification_label_schema(
        self, 
        label_groups: List[LabelCategories.LabelGroup], 
        label_entities: List[LabelEntity]
    ) -> LabelSchemaEntity:
        """ Generate LabelSchema for Classification. """
        label_schema = LabelSchemaEntity()

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
                        group_type=LabelGroupType.EXCLUSIVE
                    )
                )
            label_schema.add_group(self._generate_empty_label_entity())
        else:
            label_schema = self._generate_default_label_schema(label_entities)

        return label_schema
        
