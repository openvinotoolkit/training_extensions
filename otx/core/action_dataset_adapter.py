"""Action Classification / Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
import os
import os.path as osp
from typing import List

from datumaro.components.dataset import Dataset as DatumaroDataset
from datumaro.components.annotation import AnnotationType as DatumaroAnnotationType
from datumaro.components.annotation import Bbox as DatumaroBbox

from otx.core.base_dataset_adapter import BaseDatasetAdapter
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.label import LabelEntity
from otx.api.entities.annotation import (Annotation, AnnotationSceneEntity, AnnotationSceneKind, NullAnnotationSceneEntity)
from otx.utils.logger import get_logger

logger = get_logger()

class ActionDatasetAdapter(BaseDatasetAdapter):
    def import_dataset(
        self,
        train_data_roots: str,
        val_data_roots: str,
        test_data_roots: str = None,
        unlabeled_data_roots: str = None
    ) -> DatumaroDataset:
        self.dataset = {}
        self.dataset[Subset.TRAINING] = self._prepare_cvat_pair_data(train_data_roots)
        self.dataset[Subset.VALIDATION] = self._prepare_cvat_pair_data(val_data_roots)
        if test_data_roots:
            self.dataset[Subset.TESTING] = self._prepare_cvat_pair_data(test_data_roots)

        return self.dataset

    def _prepare_cvat_pair_data(self, path:str) -> List[DatumaroDataset]:
        """ Preparing a list of DatumaroDataset. """
        cvat_data_list = []
        for cvat_data in os.listdir(path):
            cvat_data_path = osp.join(path, cvat_data)
            cvat_data_list.append(DatumaroDataset.import_from(cvat_data_path, 'cvat'))
        return cvat_data_list
    
    def _prepare_label_information(self, datumaro_dataset: dict) -> dict:
        outputs = {
            "category_items": [],
            "label_groups" : [],
            "label_entities": [],
        }
        category_list = []
        for cvat_data in datumaro_dataset[Subset.TRAINING]:
            categories = cvat_data.categories().get(DatumaroAnnotationType.label, None)
            
            if categories not in category_list:
                outputs["category_items"].extend(categories.items)
                outputs["label_groups"].extend(categories.label_groups)
            
            category_list.append(categories)
        
        outputs["label_entities"] = [LabelEntity(name=class_name.name, domain=self.domain,
                            is_empty=False, id=ID(i)) for i, class_name in enumerate(outputs["category_items"])]
        return outputs

    def convert_to_otx_format(self, datumaro_dataset: dict) -> DatasetEntity:
        label_information = self._prepare_label_information(datumaro_dataset)
        label_entities = label_information["label_entities"]

        label_schema = self._generate_default_label_schema(label_entities)

        dataset_items = []
        for subset, subset_data in datumaro_dataset.items():
            for datumaro_items in subset_data:
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    shapes = []
                    for ann in datumaro_item.annotations:
                        if isinstance(ann, DatumaroBbox):
                            shapes.append(
                                Annotation(
                                    Rectangle(
                                        x1=ann.points[0]/image.width, 
                                        y1=ann.points[1]/image.height,
                                        x2=ann.points[2]/image.width,
                                        y2=ann.points[3]/image.height),
                                    labels = [
                                        ScoredLabel(
                                            label=label_entities[ann.label]
                                        )
                                    ]
                                )
                            )
                    # Unlabeled dataset
                    if len(shapes) == 0:
                        annotation_scene = NullAnnotationSceneEntity()
                    else:
                        annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items), label_schema