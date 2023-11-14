"""Base Class for Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import os
from abc import abstractmethod
from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Union, List

import cv2
import datumaro
import numpy as np
from datumaro.components.annotation import Annotation as DatumAnnotation
from datumaro.components.annotation import AnnotationType as DatumAnnotationType
from datumaro.components.annotation import Categories as DatumCategories
from datumaro.components.annotation import LabelCategories as DatumLabelCategories
from datumaro.components.dataset import Dataset as DatumDataset
from datumaro.components.dataset import DatasetSubset as DatumDatasetSubset
from datumaro.components.dataset import eager_mode
from datumaro.components.media import Image as DatumImage
from datumaro.components.media import MediaElement as DatumMediaElement

from otx.v2.adapters.datumaro.caching.storage_cache import init_arrow_cache
from otx.v2.api.core.dataset import BaseDatasetAdapter
from otx.v2.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
    NullAnnotationSceneEntity,
)
from otx.v2.api.entities.id import ID
from otx.v2.api.entities.image import Image
from otx.v2.api.entities.label import LabelEntity
from otx.v2.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.v2.api.entities.media import IMediaEntity
from otx.v2.api.entities.scored_label import ScoredLabel
from otx.v2.api.entities.shapes.polygon import Point, Polygon
from otx.v2.api.entities.shapes.rectangle import Rectangle
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType

LabelInformationType = Dict[str, Union[List[LabelEntity], List[DatumCategories]]]
class DatumaroDatasetAdapter(BaseDatasetAdapter):
    """Base dataset adapter for all of downstream tasks to use Datumaro.

    Mainly, DatumaroDatasetAdapter detect and import the dataset by using the function implemented in Datumaro.
    And it could prepare common variable, function (EmptyLabelSchema, LabelSchema, ..) commonly consumed under all tasks

    Args:
        task_type (TaskType): type of the task
        train_data_roots (str | None): Path for training data
        train_ann_files (str | None): Path for training annotation file
        val_data_roots (str | None): Path for validation data
        val_ann_files (str | None): Path for validation annotation file
        test_data_roots (str | None): Path for test data
        test_ann_files (str | None): Path for test annotation file
        unlabeled_data_roots (str | None): Path for unlabeled data
        unlabeled_file_list (str | None): Path of unlabeled file list
        encryption_key (str | None): Encryption key to load an encrypted dataset
                                        (only required for DatumaroBinary format)

    Since all adapters can be used for training and validation,
    the default value of train/val/test_data_roots was set to None.

    i.e)
    For the training/validation phase, test_data_roots is not used.
    For the test phase, train_data_roots and val_data_root are not used.
    """

    def __init__(
        self,
        task_type: TaskType,
        train_data_roots: str | None = None,
        train_ann_files: str | None = None,
        val_data_roots: str | None = None,
        val_ann_files: str | None = None,
        test_data_roots: str | None = None,
        test_ann_files: str | None = None,
        unlabeled_data_roots: str | None = None,
        unlabeled_file_list: str | None = None,
        cache_config: dict | None = None,
        encryption_key: str | None = None,
        use_mask: bool = False,
        **kwargs,
    ) -> None:
        self.task_type = task_type
        self.domain = task_type.domain
        self.data_type: str
        self.is_train_phase: bool
        self.use_mask = use_mask

        self.dataset = self._import_datasets(
            train_data_roots=train_data_roots,
            train_ann_files=train_ann_files,
            val_data_roots=val_data_roots,
            val_ann_files=val_ann_files,
            test_data_roots=test_data_roots,
            test_ann_files=test_ann_files,
            unlabeled_data_roots=unlabeled_data_roots,
            unlabeled_file_list=unlabeled_file_list,
            encryption_key=encryption_key,
            **kwargs,
        )

        cache_config = cache_config if cache_config is not None else {}
        for subset, dataset in self.dataset.items():
            # cache these subsets only
            if subset in (Subset.TRAINING, Subset.VALIDATION, Subset.UNLABELED, Subset.PSEUDOLABELED):
                self.dataset[subset] = init_arrow_cache(dataset, **cache_config)
        
        label_information: LabelInformationType = self._prepare_label_information(self.dataset)
        
        self.label_entities: list[LabelEntity] = label_information["label_entities"]  
        self.category_items: list[DatumCategories] = label_information["category_items"]
        self.label_groups: list[DatumLabelCategories] = label_information["label_groups"]
        
        self.label_schema: LabelSchemaEntity

    def _import_datasets(
        self,
        train_data_roots: str | None = None,
        train_ann_files: str | None = None,
        val_data_roots: str | None = None,
        val_ann_files: str | None = None,
        test_data_roots: str | None = None,
        test_ann_files: str | None = None,
        unlabeled_data_roots: str | None = None,
        unlabeled_file_list: str | None = None,
        encryption_key: str | None = None,
    ) -> dict[Subset, DatumDataset]:
        """Import datasets by using Datumaro.import_from() method.

        Args:
            train_data_roots (str | None): Path for training data
            train_ann_files (str | None): Path for training annotation files
            val_data_roots (str | None): Path for validation data
            val_ann_files (str | None): Path for validation annotation files
            test_data_roots (str | None): Path for test data
            test_ann_files (str | None): Path for test annotation files
            unlabeled_data_roots (str | None): Path for unlabeled data
            unlabeled_file_list (str | None): Path for unlabeled file list
            encryption_key (str | None): Encryption key to load an encrypted dataset
                                            (DatumaroBinary format)

        Returns:
            DatumDataset: Datumaro Dataset
        """
        dataset = {}
        if train_data_roots is None and val_data_roots is None and test_data_roots is None:
            raise ValueError("At least 1 data_root is needed to train/test.")

        # Construct dataset for training, validation, testing, unlabeled
        if train_data_roots is not None:
            train_dataset = self._import_dataset(train_data_roots, train_ann_files, encryption_key, Subset.TRAINING)
            dataset[Subset.TRAINING] = self._get_subset_data("train", train_dataset)

            # If validation is manually defined --> set the validation data according to user's input
            if val_data_roots:
                val_dataset = self._import_dataset(val_data_roots, val_ann_files, encryption_key, Subset.VALIDATION)
                dataset[Subset.VALIDATION] = self._get_subset_data("val", val_dataset)
            elif "val" in train_dataset.subsets():
                dataset[Subset.VALIDATION] = self._get_subset_data("val", train_dataset)

        if test_data_roots is not None:
            test_dataset = self._import_dataset(test_data_roots, test_ann_files, encryption_key, Subset.TESTING)
            dataset[Subset.TESTING] = self._get_subset_data("test", test_dataset)

        if unlabeled_data_roots is not None:
            dataset[Subset.UNLABELED] = DatumDataset.import_from(unlabeled_data_roots, format="image_dir")
            if unlabeled_file_list is not None:
                self._filter_unlabeled_data(dataset[Subset.UNLABELED], unlabeled_file_list)
        return dataset

    def _import_dataset(
        self,
        data_roots: str | None,
        ann_files: str | None,
        encryption_key: str | None,
        mode: Subset,
    ) -> DatumDataset:
        # Find self.data_type and task_type
        mode_to_str = {Subset.TRAINING: "train", Subset.VALIDATION: "val", Subset.TESTING: "test"}
        str_mode = mode_to_str[mode]

        if data_roots is not None:
            data_path = Path(data_roots)
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset path '{data_roots}' does not exist.")
            data_roots = str(data_path)
        self.data_type_candidates = self._detect_dataset_format(path=data_roots)
        self.data_type = self._select_data_type(self.data_type_candidates)

        dataset_kwargs: dict = {"path": data_roots, "format": self.data_type}
        if ann_files is not None:
            if self.data_type not in ("coco"):
                raise NotImplementedError(
                    f"Specifying '--{str_mode}-ann-files' is not supported for data type '{self.data_type}'",
                )
            dataset_kwargs["path"] = ann_files
            dataset_kwargs["subset"] = str_mode

        if encryption_key is not None:
            dataset_kwargs["encryption_key"] = encryption_key

        if self.task_type == TaskType.VISUAL_PROMPTING and self.data_type in ["coco"]:
            dataset_kwargs["merge_instance_polygons"] = self.use_mask

        dataset = DatumDataset.import_from(**dataset_kwargs)

        return dataset

    @abstractmethod
    def get_otx_dataset(self) -> dict[Subset, DatumDataset]:
        """Get DatasetEntity."""
        raise NotImplementedError

    def get_label_schema(self) -> LabelSchemaEntity:
        """Get Label Schema."""
        return self._generate_default_label_schema(self.label_entities)

    def _get_subset_data(self, subset: str, dataset: DatumDataset) -> DatumDatasetSubset:
        """Get subset dataset according to subset."""
        with eager_mode(True, dataset):
            subsets = list(dataset.subsets().keys())

            for s in [subset, "default"]:
                _s = s
                if subset == "val" and _s != "default":
                    _s = "valid"
                exact_subset = get_close_matches(_s, subsets, cutoff=0.5)

                if exact_subset:
                    return dataset.subsets()[exact_subset[0]].as_dataset()
                elif subset == "test":
                    # If there is not test dataset in data.yml, then validation set will be test dataset
                    _s = "valid"
                    exact_subset = get_close_matches(_s, subsets, cutoff=0.5)
                    if exact_subset:
                        return dataset.subsets()[exact_subset[0]].as_dataset()

        raise ValueError("Can't find proper dataset.")

    def _detect_dataset_format(self, path: str | None) -> str:
        """Detect dataset format (ImageNet, COCO, ...)."""
        return datumaro.Environment().detect_dataset(path=path)

    def _generate_empty_label_entity(self) -> LabelGroup:
        """Generate Empty Label Group for H-label, Multi-label Classification."""
        empty_label = LabelEntity(name="Empty label", is_empty=True, domain=self.domain)
        empty_group = LabelGroup(name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL)
        return empty_group

    def _generate_default_label_schema(self, label_entities: list[LabelEntity]) -> LabelSchemaEntity:
        """Generate Default Label Schema for Multi-class Classification, Detecion, Etc."""
        label_schema = LabelSchemaEntity()
        main_group = LabelGroup(
            name="labels",
            labels=label_entities,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        label_schema.add_group(main_group)
        return label_schema

    def _prepare_label_information(
        self,
        datumaro_dataset: dict[Subset, DatumDataset],
    ) -> LabelInformationType:
        # Get datumaro category information
        if Subset.TRAINING in datumaro_dataset:
            label_categories_list = datumaro_dataset[Subset.TRAINING].categories().get(DatumAnnotationType.label, None)
        elif Subset.VALIDATION in datumaro_dataset:
            label_categories_list = (
                datumaro_dataset[Subset.VALIDATION].categories().get(DatumAnnotationType.label, None)
            )
        else:
            label_categories_list = datumaro_dataset[Subset.TESTING].categories().get(DatumAnnotationType.label, None)
        category_items = label_categories_list.items if label_categories_list is not None else []
        label_groups = label_categories_list.label_groups if label_categories_list is not None else []

        # LabelEntities
        label_entities = [
            LabelEntity(name=class_name.name, domain=self.domain, is_empty=False, id=ID(i))
            for i, class_name in enumerate(category_items)
        ]

        return {"category_items": category_items, "label_groups": label_groups, "label_entities": label_entities}

    def _is_normal_polygon(self, annotation: DatumAnnotationType.polygon) -> bool:
        """To filter out the abnormal polygon."""
        x_points = [annotation.points[i] for i in range(0, len(annotation.points), 2)]
        y_points = [annotation.points[i + 1] for i in range(0, len(annotation.points), 2)]
        return min(x_points) < max(x_points) and min(y_points) < max(y_points)

    def _is_normal_bbox(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """To filter out the abrnormal bbox."""
        return x1 < x2 and y1 < y2

    def _select_data_type(self, data_candidates: list[str] | str) -> str:
        """Select specific type among candidates.

        Args:
            data_candidates (Union[List[str], str]): Type candidates made by Datumaro.Environment().detect_dataset()

        Returns:
            str: Selected data type
        """
        return data_candidates[0]

    def _get_ann_scene_entity(self, shapes: list[Annotation]) -> AnnotationSceneEntity:
        annotation_scene: AnnotationSceneEntity | None = None
        if len(shapes) == 0:
            annotation_scene = NullAnnotationSceneEntity()
        else:
            annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
        return annotation_scene

    def _get_label_entity(self, annotation: DatumAnnotation) -> Annotation:
        """Get label entity."""
        return Annotation(
            Rectangle.generate_full_box(),
            labels=[ScoredLabel(label=self.label_entities[annotation.label])],
        )

    def _get_normalized_bbox_entity(self, annotation: DatumAnnotation, width: int, height: int) -> Annotation:
        """Get bbox entity w/ normalization."""
        x1, y1, x2, y2 = annotation.points
        return Annotation(
            Rectangle(
                x1=x1 / width,
                y1=y1 / height,
                x2=x2 / width,
                y2=y2 / height,
            ),
            labels=[ScoredLabel(label=self.label_entities[annotation.label])],
        )

    def _get_original_bbox_entity(self, annotation: DatumAnnotation) -> Annotation:
        """Get bbox entity w/o normalization."""
        return Annotation(
            Rectangle(
                x1=annotation.points[0],
                y1=annotation.points[1],
                x2=annotation.points[2],
                y2=annotation.points[3],
            ),
            labels=[ScoredLabel(label=self.label_entities[annotation.label])],
        )

    def _get_polygon_entity(
        self,
        annotation: DatumAnnotation,
        width: int,
        height: int,
        num_polygons: int = -1,
    ) -> Annotation:
        """Get polygon entity."""
        polygon = Polygon(
            points=[
                Point(x=annotation.points[i] / width, y=annotation.points[i + 1] / height)
                for i in range(0, len(annotation.points), 2)
            ],
        )
        step = 1 if num_polygons == -1 else len(polygon.points) // num_polygons
        points = [polygon.points[i] for i in range(0, len(polygon.points), step)]

        return Annotation(
            Polygon(points),
            labels=[ScoredLabel(label=self.label_entities[annotation.label])],
        )

    def _get_mask_entity(self, annotation: DatumAnnotation) -> Annotation:
        """Get mask entity."""
        mask = Image(data=annotation.image, size=annotation.image.shape)
        return Annotation(
            mask,
            labels=[ScoredLabel(label=self.label_entities[annotation.label])],
        )

    def remove_unused_label_entities(self, used_labels: list) -> None:
        """Remove unused label from label entities.

        Because label entities will be used to make Label Schema,
        If there is unused label in Label Schema, it will hurts the model performance.
        So, remove the unused label from label entities.

        Args:
            used_labels (List): list for index of used label
        """
        clean_label_entities = []
        for used_label in used_labels:
            clean_label_entities.append(self.label_entities[used_label])
        self.label_entities = clean_label_entities

    def _filter_unlabeled_data(self, unlabeled_dataset: DatumDataset, unlabeled_file_list: str) -> None:
        """Filter out unlabeled dataset which isn't included in unlabeled file list."""
        allowed_extensions = ["jpg", "png", "jpeg"]
        file_list = []
        with open(unlabeled_file_list, encoding="utf-8") as f:
            for line in f.readlines():
                file_ext = line.rstrip().split(".")[-1]
                file_list.append(line.split(".")[0])

                if file_ext.lower() not in allowed_extensions:
                    raise ValueError(f"{file_ext} is not supported type for unlabeled data.")

        copy_dataset = deepcopy(unlabeled_dataset)
        for item in copy_dataset:
            if item.id not in file_list:
                unlabeled_dataset.remove(item.id, item.subset)

    @staticmethod
    def datum_media_2_otx_media(datumaro_media: DatumMediaElement) -> IMediaEntity:
        """Convert Datumaro media to OTX media."""
        if isinstance(datumaro_media, DatumImage):
            path = getattr(datumaro_media, "path", None)
            size = datumaro_media._size

            if path and os.path.exists(path) and not datumaro_media.is_encrypted:
                return Image(file_path=path, size=size)

            def helper() -> cv2.typing.MatLike:
                data = datumaro_media.data
                # OTX expects unint8 data type
                data = data.astype(np.uint8)
                # OTX expects RGB format
                if len(data.shape) == 2:
                    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
                if len(data.shape) == 3:
                    if data.shape[-1] == 3:
                        return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    if data.shape[-1] == 4:
                        return cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)
                raise NotImplementedError

            return Image(data=helper, size=size)
        raise NotImplementedError
