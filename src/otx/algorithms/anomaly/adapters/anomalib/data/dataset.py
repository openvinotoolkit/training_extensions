"""DataLoaders for Anomaly Tasks."""
# Copyright (C) 2021 Intel Corporation
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

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from bson import ObjectId

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset


class BaseAnomalyDataset(DatasetEntity, ABC):
    """Base Dataloader for Anomaly Tasks."""

    def __init__(
        self,
        train_subset: Optional[Dict[str, str]] = None,
        val_subset: Optional[Dict[str, str]] = None,
        test_subset: Optional[Dict[str, str]] = None,
    ):
        """Base Anomaly Dataset.

        Args:
            train_subset (Optional[Dict[str, str]], optional): Path to annotation
                and dataset used for training. Defaults to None.
            val_subset (Optional[Dict[str, str]], optional): Path to annotation
                and dataset used for validation. Defaults to None.
            test_subset (Optional[Dict[str, str]], optional): Path to annotation
                and dataset used for testing. Defaults to None.
        """
        items: List[DatasetItemEntity] = []
        self.normal_label = LabelEntity(id=ID(0), name="Normal", domain=Domain.ANOMALY_CLASSIFICATION)
        self.abnormal_label = LabelEntity(
            id=ID(1),
            name="Anomalous",
            domain=Domain.ANOMALY_CLASSIFICATION,
            is_anomalous=True,
        )

        if train_subset is not None:
            train_ann_file = Path(train_subset["ann_file"])
            train_data_root = Path(train_subset["data_root"])
            items.extend(
                self.get_dataset_items(
                    ann_file_path=train_ann_file,
                    data_root_dir=train_data_root,
                    subset=Subset.TRAINING,
                )
            )

        if val_subset is not None:
            val_ann_file = Path(val_subset["ann_file"])
            val_data_root = Path(val_subset["data_root"])
            items.extend(
                self.get_dataset_items(
                    ann_file_path=val_ann_file,
                    data_root_dir=val_data_root,
                    subset=Subset.VALIDATION,
                )
            )

        if test_subset is not None:
            test_ann_file = Path(test_subset["ann_file"])
            test_data_root = Path(test_subset["data_root"])
            items.extend(
                self.get_dataset_items(
                    ann_file_path=test_ann_file,
                    data_root_dir=test_data_root,
                    subset=Subset.TESTING,
                )
            )

        super().__init__(items=items)

    @abstractmethod
    def get_dataset_items(self, ann_file_path: Path, data_root_dir: Path, subset: Subset) -> List[DatasetItemEntity]:
        """To be implemented ib subclasses."""
        raise NotImplementedError


class AnomalyClassificationDataset(BaseAnomalyDataset):
    """Dataloader for Anomaly Classification Task.

    Example:
    >>> train_subset = {
            "ann_file": "tests/assets/anomaly/classification/train.json",
            "data_root": "tests/assets/anomaly/hazelnut",
        }
    >>> val_subset = {
            "ann_file": "tests/assets/anomaly/classification/val.json",
            "data_root": "tests/assets/anomaly/hazelnut"
        }
    >>> training_dataset = AnomalyClassificationDataset(
            train_subset=train_subset, val_subset=val_subset
        )
    >>> test_subset = {
            "ann_file": "tests/assets/anomaly/classification/test.json",
            "data_root": "tests/assets/anomaly/hazelnut"
        }
    >>> testing_dataset = AnomalyClassificationDataset(test_subset=test_subset)
    """

    def get_dataset_items(self, ann_file_path: Path, data_root_dir: Path, subset: Subset) -> List[DatasetItemEntity]:
        """Loads dataset based on the image path in annotation file.

        Args:
            ann_file_path (Path): Path to json containing the annotations.
                For example of annotation look at `tests/assets/anomaly/[train, test,val].json.
            data_root_dir (Path): Path to folder containing images.
            subset (Subset): Subset of the dataset.

        Returns:
            List[DatasetItemEntity]: List containing subset dataset.
        """
        # read annotation file
        samples = pd.read_json(ann_file_path)

        dataset_items = []
        for _, sample in samples.iterrows():
            # Create image
            # convert path to str as PosixPath is not supported by Image
            image = Image(file_path=str(data_root_dir / sample.image_path))
            # Create annotation
            shape = Rectangle.generate_full_box()
            label: LabelEntity = self.normal_label if sample.label == "good" else self.abnormal_label
            labels = [ScoredLabel(label, probability=1.0)]
            annotations = [Annotation(shape=shape, labels=labels)]
            annotation_scene = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)

            # Create dataset item
            dataset_item = DatasetItemEntity(media=image, annotation_scene=annotation_scene, subset=subset)
            # Add to dataset items
            dataset_items.append(dataset_item)

        return dataset_items


class AnomalySegmentationDataset(BaseAnomalyDataset):
    """Dataloader for Anomaly Segmentation Task.

    Example:
        >>> train_subset = {
                "ann_file": "tests/assets/anomaly/segmentation/train.json",
                "data_root": "tests/assets/anomaly/hazelnut",
            }
        >>> val_subset = {
                "ann_file": "tests/assets/anomaly/segmentation/val.json",
                "data_root": "tests/assets/anomaly/hazelnut"
            }
        >>> training_dataset = AnomalySegmentationDataset(
                train_subset=train_subset, val_subset=val_subset
            )
        >>> test_subset = {
                "ann_file": "tests/assets/anomaly/segmentation/test.json",
                "data_root": "tests/assets/anomaly/hazelnut"
            }
        >>> testing_dataset = AnomalySegmentationDataset(test_subset=test_subset)

    """

    def get_dataset_items(self, ann_file_path: Path, data_root_dir: Path, subset: Subset) -> List[DatasetItemEntity]:
        """Loads dataset based on the image path in annotation file.

        Args:
            ann_file_path (Path): Path to json containing the annotations.
                For example of annotation look at `tests/assets/anomaly/[train, test,val].json.
            data_root_dir (Path): Path to folder containing images.
            subset (Subset): Subset of the dataset.

        Returns:
            List[DatasetItemEntity]: List containing subset dataset.
        """
        # read annotation file
        samples = pd.read_json(ann_file_path)

        dataset_items = []
        for _, sample in samples.iterrows():
            # Create image
            # convert path to str as PosixPath is not supported by Image
            image = Image(file_path=str(data_root_dir / sample.image_path))
            # Create annotation
            label: LabelEntity = self.normal_label if sample.label == "good" else self.abnormal_label
            annotations = [
                Annotation(
                    Rectangle.generate_full_box(),
                    labels=[ScoredLabel(label=label, probability=1.0)],
                )
            ]
            if isinstance(sample.masks, list) and len(sample.masks) > 0:
                for contour in sample.masks:
                    points = [Point(x, y) for x, y in contour]
                    polygon = Polygon(points=points)
                    if polygon.get_area() > 0:
                        # Contour is a closed polygon with area > 0
                        annotations.append(
                            Annotation(
                                shape=polygon,
                                labels=[ScoredLabel(label, 1.0)],
                                id=ID(ObjectId()),
                            )
                        )
                    else:
                        # Contour is a closed polygon with area == 0
                        warnings.warn(
                            "The geometry of the segmentation map you are converting "
                            "is not fully supported. Polygons with a area of zero "
                            "will be removed.",
                            UserWarning,
                        )
            annotation_scene = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)

            # Add to dataset items
            dataset_items.append(DatasetItemEntity(media=image, annotation_scene=annotation_scene, subset=subset))

        return dataset_items


class AnomalyDetectionDataset(BaseAnomalyDataset):
    """Dataloader for Anomaly Segmentation Task.

    Example:
        >>> train_subset = {
                "ann_file": "tests/assets/anomaly/detection/train.json",
                "data_root": "tests/assets/anomaly/hazelnut",
            }
        >>> val_subset = {
                "ann_file": "tests/assets/anomaly/detection/val.json",
                "data_root": "tests/assets/anomaly/hazelnut"
            }
        >>> training_dataset = AnomalyDetectionDataset(
                train_subset=train_subset, val_subset=val_subset
            )
        >>> test_subset = {
                "ann_file": "tests/assets/anomaly/detection/test.json",
                "data_root": "tests/assets/anomaly/hazelnut"
            }
        >>> testing_dataset = AnomalyDetectionDataset(test_subset=test_subset)

    """

    def get_dataset_items(self, ann_file_path: Path, data_root_dir: Path, subset: Subset) -> List[DatasetItemEntity]:
        """Loads dataset based on the image path in annotation file.

        Args:
            ann_file_path (Path): Path to json containing the annotations.
                For example of annotation look at `tests/assets/anomaly/[train, test,val].json.
            data_root_dir (Path): Path to folder containing images.
            subset (Subset): Subset of the dataset.

        Returns:
            List[DatasetItemEntity]: List containing subset dataset.
        """
        # read annotation file
        samples = pd.read_json(ann_file_path)

        dataset_items = []
        for _, sample in samples.iterrows():
            # Create image
            # convert path to str as PosixPath is not supported by Image
            image = Image(file_path=str(data_root_dir / sample.image_path))
            # Create annotation
            label: LabelEntity = self.normal_label if sample.label == "good" else self.abnormal_label
            annotations = [
                Annotation(
                    Rectangle.generate_full_box(),
                    labels=[ScoredLabel(label=label, probability=1.0)],
                )
            ]
            if isinstance(sample.bboxes, list) and len(sample.bboxes) > 0:
                for bbox in sample.bboxes:
                    box = Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                    if box.get_area() > 0:
                        # Contour is a closed polygon with area > 0
                        annotations.append(
                            Annotation(
                                shape=box,
                                labels=[ScoredLabel(label, 1.0)],
                                id=ID(ObjectId()),
                            )
                        )
                    else:
                        # Contour is a closed polygon with area == 0
                        warnings.warn(
                            "The geometry of the segmentation map you are converting "
                            "is not fully supported. Polygons with a area of zero "
                            "will be removed.",
                            UserWarning,
                        )
            annotation_scene = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)

            # Add to dataset items
            dataset_items.append(DatasetItemEntity(media=image, annotation_scene=annotation_scene, subset=subset))

        return dataset_items
