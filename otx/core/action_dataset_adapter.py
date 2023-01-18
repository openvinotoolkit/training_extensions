"""Action Base / Classification / Detection Dataset Adapter."""

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
from datumaro.components.annotation import Label as DatumaroLabel

from otx.core.base_dataset_adapter import BaseDatasetAdapter
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.metadata import MetadataItemEntity, FloatMetadata
from otx.api.entities.subset import Subset
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.label import LabelEntity
from otx.api.entities.annotation import (Annotation, AnnotationSceneEntity, AnnotationSceneKind, NullAnnotationSceneEntity)
from otx.utils.logger import get_logger

logger = get_logger()

class ActionBaseDatasetAdapter(BaseDatasetAdapter):
    def import_dataset(
        self,
        train_data_roots: str,
        val_data_roots: str,
        test_data_roots: str = None,
        unlabeled_data_roots: str = None
    ) -> DatumaroDataset:
        """ Import multiple videos that have CVAT format annotation. """
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
        """ Prepare and reorganize the label information for merging multiple video information.
        
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
            "label_groups" : [],
            "label_entities": [],
        }

        category_list = [] # to check the duplicate case
        for cvat_data in datumaro_dataset[Subset.TRAINING]:
            # Making overall categories
            categories = cvat_data.categories().get(DatumaroAnnotationType.label, None)
            
            if categories not in category_list:
                outputs["category_items"].extend(categories.items)
                outputs["label_groups"].extend(categories.label_groups)
            
            category_list.append(categories)

            # Reindexing the each label index of multiple video datasets
            for cvat_data_item in cvat_data:
                for ann in cvat_data_item.annotations:
                    ann_name = categories.items[ann.label].name
                    ann.label = [i for i, category_item in enumerate(outputs["category_items"]) if category_item.name == ann_name][0]

        # Generate label_entity list according to overall categories 
        outputs["label_entities"] = [LabelEntity(name=class_name.name, domain=self.domain,
                            is_empty=False, id=ID(i)) for i, class_name in enumerate(outputs["category_items"])]
        return outputs

class ActionClassificationDatasetAdapter(ActionBaseDatasetAdapter, BaseDatasetAdapter):
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
                        # Action Classification
                        if isinstance(ann, DatumaroLabel):
                            shapes.append(
                                Annotation(
                                    Rectangle.generate_full_box(),
                                    labels = [
                                        ScoredLabel(
                                            label=label_entities[ann.label]                                        )
                                    ]
                                )
                            )
                    meta = MetadataItemEntity(
                        data=
                    )
                    # Unlabeled dataset
                    if len(shapes) == 0:
                        annotation_scene = NullAnnotationSceneEntity()
                    else:
                        annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)
        return DatasetEntity(items=dataset_items), label_schema

class ActionDetectionDatasetAdapter(ActionBaseDatasetAdapter, BaseDatasetAdapter):
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