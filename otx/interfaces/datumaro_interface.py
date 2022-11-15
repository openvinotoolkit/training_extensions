"""Interface for Datumaro integration."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

# pylint: disable=too-many-nested-blocks, invalid-name
import datumaro
from datumaro.components.dataset import Dataset as DatumaroDataset
from datumaro.components.annotation import Bbox as DatumaroBbox
from datumaro.components.annotation import Mask as DatumaroMask
from datumaro.plugins.transforms import MasksToPolygons
from datumaro.components.annotation import AnnotationType as DatumaroAnnotationType

from otx.api.entities.annotation import (Annotation, AnnotationSceneEntity, AnnotationSceneKind)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import (LabelGroup, LabelGroupType, LabelSchemaEntity)
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.subset import Subset
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

###### Procedure ######
# 1. Find the type of dataset by using Datumaro.Environment.detection_datset()
# 1.1 Excpetion handling for unsupported dataset type
# Classification : ImageNet(like ImageDirectory used Torchvision)
# Detection : VOC, COCO
# Segmentation : Common, CityScape, ADE20K
# Action : CVAT

# 2. Converting process (Datumaro Dataset --> DatasetEntity(OTE_SDK))

# TODOs
# TODO1: Consider H-label, multi-label classification (need to more analysis about Datumaro)
# TODO2: Consider auto-split by using Datumaro function
# TODO3: Unlabeled support 
class DatumaroHandler:
    """Handler to use Datumaro as a front-end dataset."""
    def __init__(self):
        self.supported_task_data_dict = {
            'classification': ['imagenet'],
            'detection': ['coco', 'voc'], 
            'segmentation': ['common_semantic_segmentation', 'voc', 'cityscapes', 'ade20k2017', 'ade20k2020'],
            'common': ['cvat'] #TODO: consider cvat for only video? 
        }
    
    def import_dataset(
            self,
            train_data_roots: str,
            train_ann_files: str = None,
            val_data_roots: str = None,
            val_ann_files: str = None,
            test_data_roots: str = None,
            test_ann_files: str = None,
            unlabeled_data_roots: str = None,
            unlabeled_file_lists: float = None
        )-> DatumaroDataset:
        """ Import dataset by using Datumaro."""
        # Find self.data_type and task_type
        data_type_candidates = self._detect_dataset_format(path=train_data_roots)
        print('[*] data_type_candidates: ', data_type_candidates)
        
        #TODO: more better way for classification
        if 'imagenet' in data_type_candidates:
            self.data_type = 'imagenet'
        else:
            self.data_type = data_type_candidates[0]
        print('[*] selected data type: ', self.data_type)
        
        self.task_type = self._find_task_type(self.data_type)
        print('[*] task_type: ', self.task_type)
        self._set_domain(self.task_type)

        # Construct dataset for training, validation, unlabeled
        self.dataset = {}
        datumaro_dataset = DatumaroDataset.import_from(train_data_roots, format=self.data_type)

        # Annotation type filtering
        # TODO: is there something better? 
        if DatumaroAnnotationType.mask in list(datumaro_dataset.categories().keys()):
            datumaro_dataset.categories().pop(DatumaroAnnotationType.mask)

        # Prepare subsets by using Datumaro dataset
        for k, v in datumaro_dataset.subsets().items():
            if 'train' in k or 'default' in k:
                self.dataset[Subset.TRAINING] = v
            elif 'val' in k or 'test' in k:
                self.dataset[Subset.VALIDATION] = v
        
        # If validation is manually defined --> set the validation data according to user's input
        if val_data_roots is not None:
            val_data_type = self._detect_dataset_format(path=val_data_roots)
            assert self.data_type == val_data_type, "The data types of training and validation must be same"
            self.dataset[Subset.VALIDATION] = DatumaroDataset.import_from(val_data_roots, format=val_data_type)

        if Subset.VALIDATION not in self.dataset.keys():
            #TODO: auto_split
            pass

        # If unlabeled data is defined --> Semi-SL enable?
        if unlabeled_data_roots is not None:
            self.dataset[Subset.UNLABELED] = DatumaroDataset.import_from(unlabeled_data_roots, format='image_dir')
            #TODO: enable to read unlabeled file lists
        
        return self.dataset
    
    def _find_task_type(self, data_type:str ) -> str:
        """ Find task type (cls, det, seg) by using data type. """
        for k in self.supported_task_data_dict.keys():
            if self.data_type in self.supported_task_data_dict[k]:
                return k
        return ValueError("{} is not supported data type, supported data type: {}".format(self.data_type, self.supported_task_data_dict.values()))

    def _auto_split(self): ## To be implemented
        """ Automatic train/val split."""
        return

    def _detect_dataset_format(self, path: str) -> str:
        """ Detection dataset format (ImageNet, COCO, Cityscapes, ...). """
        return datumaro.Environment().detect_dataset(path=path) 

    def _set_domain(self, task_type: str) -> None:
        """ Get domain type."""
        if task_type == 'classification':
            self.domain = Domain.CLASSIFICATION
        elif task_type == 'detection':
            self.domain = Domain.DETECTION
        elif task_type == 'segmentation':
            self.domain = Domain.SEGMENTATION
        elif task_type == 'video':
            raise NotImplementedError()
        else:
            raise ValueError("{} is not proper type of task, supported task type: {}".format(
                task_type, self.supported_task_data_dict.keys()))
    

    def convert_to_otx_format(self, datumaro_dataset:dict) -> DatasetEntity:
        """ Convert Datumaro Datset to DatasetEntity(OTE_SDK)"""
        dataset_items = []
        for subset, subset_data in datumaro_dataset.items():
            class_name_items = list(subset_data.categories().values())[-1].items
            label_entities = [LabelEntity(name=class_name.name, domain=self.domain,
                                is_empty=False, id=ID(i)) for i, class_name in enumerate(class_name_items)]
            for phase, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    print('[*] datumaro_item: ', datumaro_item)
                    image = Image(data=datumaro_item.media.data)
                    if self.domain == Domain.CLASSIFICATION:
                        labels = [
                            ScoredLabel(
                                label= [label for label in label_entities if label.name == class_name_items[ann.label].name][0],
                                probability=1.0   
                            ) for ann in datumaro_item.annotations
                        ]
                        shapes = [Annotation(Rectangle.generate_full_box(), labels)]
                    
                    elif self.domain == Domain.DETECTION:
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
                    elif self.domain == Domain.SEGMENTATION:
                        shapes = []
                        for ann in datumaro_item.annotations:
                            if isinstance(ann, DatumaroMask):
                                datumaro_polygons = MasksToPolygons.convert_mask(ann)
                                for d_polygon in datumaro_polygons:
                                    shapes.append(
                                        Annotation(
                                            Polygon(points=[Point(x=d_polygon.points[i]/image.width,y=d_polygon.points[i+1]/image.height) for i in range(len(d_polygon.points)-1)]),
                                            labels=[
                                                ScoredLabel(
                                                    label=label_entities[d_polygon.label]
                                                )
                                            ]
                                        )
                                    )
                    else : #Video
                        raise NotImplementedError()
                    annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items)
