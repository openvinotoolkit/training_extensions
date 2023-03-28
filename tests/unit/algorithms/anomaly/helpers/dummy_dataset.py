"""Datasets for testing anomaly tasks"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from bson import ObjectId
from omegaconf import OmegaConf
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader

from otx.algorithms.anomaly.adapters.anomalib.data.data import (
    OTXAnomalyDataModule,
    OTXAnomalyDataset,
)
from otx.algorithms.anomaly.adapters.anomalib.data.dataset import (
    AnomalyClassificationDataset,
    AnomalyDetectionDataset,
    AnomalySegmentationDataset,
)
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity, DatasetPurpose
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset


def get_hazelnut_dataset(task_type: TaskType, one_each: bool = False) -> DatasetEntity:
    """Get hazelnut dataset.

    Args:
        task_type (TaskType): Task type.
        one_each (bool): If this flag is true then it will sample one normal and one abnormal image for each split.
            The training split will have only one normal image. Defaults to False.
    """
    dataset: DatasetEntity
    if task_type == TaskType.ANOMALY_CLASSIFICATION:
        train_subset, test_subset, val_subset = _get_annotations("classification")
        dataset = AnomalyClassificationDataset(train_subset, val_subset, test_subset)
    elif task_type == TaskType.ANOMALY_SEGMENTATION:
        train_subset, test_subset, val_subset = _get_annotations("segmentation")
        dataset = AnomalySegmentationDataset(train_subset, val_subset, test_subset)
    elif task_type == TaskType.ANOMALY_DETECTION:
        train_subset, test_subset, val_subset = _get_annotations("detection")
        dataset = AnomalyDetectionDataset(train_subset, val_subset, test_subset)
    else:
        raise ValueError(f"{task_type} not supported.")

    if one_each:
        train_subset = []
        test_subset = []
        val_subset = []
        for item in dataset._items:
            if item.subset == Subset.TRAINING and len(train_subset) < 1:
                train_subset.append(item)
            # Check if the label is already present in the subset.
            elif item.subset == Subset.TESTING and item.annotation_scene.annotations[0].get_labels()[0].name not in [
                a.annotation_scene.annotations[0].get_labels()[0].name for a in test_subset
            ]:
                test_subset.append(item)
            # Check if the label is already present in the subset.
            elif item.subset == Subset.VALIDATION and item.annotation_scene.annotations[0].get_labels()[0].name not in [
                a.annotation_scene.annotations[0].get_labels()[0].name for a in val_subset
            ]:
                val_subset.append(item)
        dataset._items = train_subset + test_subset + val_subset
    return dataset


class DummyDataset(OTXAnomalyDataset):
    def __init__(self, task_type: TaskType):
        self.normal_label = LabelEntity(id=ID(0), name="Normal", domain=Domain.ANOMALY_CLASSIFICATION)
        self.abnormal_label = LabelEntity(
            id=ID(1),
            name="Anomalous",
            domain=Domain.ANOMALY_CLASSIFICATION,
            is_anomalous=True,
        )
        self.dataset = self.get_mock_dataitems()
        self.task_type = task_type
        self.transform = A.Compose(
            [
                A.Resize(32, 32, always_apply=True),
                ToTensorV2(),
            ]
        )
        self.config = OmegaConf.create({"dataset": {"image_size": [32, 32]}})

    def get_mock_dataitems(self) -> DatasetEntity:
        dataset_items = []

        image_anomalous = Image(np.ones((32, 32, 1)))
        annotations = [
            Annotation(Rectangle.generate_full_box(), labels=[ScoredLabel(label=self.abnormal_label, probability=1.0)])
        ]
        polygon = Polygon(points=[Point(0.0, 0.0), Point(1.0, 1.0)])
        annotations.append(
            Annotation(shape=polygon, labels=[ScoredLabel(self.abnormal_label, probability=1.0)], id=ID(ObjectId()))
        )
        annotation_scene = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)
        dataset_items.append(DatasetItemEntity(media=image_anomalous, annotation_scene=annotation_scene))

        image_normal = Image(np.zeros((32, 32, 1)))
        annotations = [
            Annotation(Rectangle.generate_full_box(), labels=[ScoredLabel(label=self.normal_label, probability=1.0)])
        ]
        annotation_scene = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)
        dataset_items.append(DatasetItemEntity(media=image_normal, annotation_scene=annotation_scene))

        return DatasetEntity(items=dataset_items, purpose=DatasetPurpose.INFERENCE)


class DummyDataModule(LightningDataModule):
    def __init__(self, task_type: TaskType):
        super().__init__()
        self.dataset = DummyDataset(task_type)
        self.labels = [self.dataset.abnormal_label, self.dataset.normal_label]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, pin_memory=True)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, shuffle=False, pin_memory=True)


class HazelnutDataModule(OTXAnomalyDataModule):
    """Creates datamodule with hazelnut dataset.

    Args:
        task_type (TaskType): Task type (classification, detection, segmentation)
    """

    def __init__(self, task_type: TaskType):
        self.config = OmegaConf.create(
            {
                "dataset": {
                    "eval_batch_size": 32,
                    "train_batch_size": 32,
                    "test_batch_size": 32,
                    "num_workers": 2,
                    "image_size": [32, 32],
                    "transform_config": {"train": None},
                }
            }
        )
        self.dataset = get_hazelnut_dataset(task_type)
        super().__init__(config=self.config, dataset=self.dataset, task_type=task_type)


def _get_annotations(task: str) -> Tuple[Dict, Dict, Dict]:
    ann_file_root = Path("tests", "assets", "anomaly", task)
    data_root = Path("tests", "assets", "anomaly", "hazelnut")

    train_subset = {"ann_file": str(ann_file_root / "train.json"), "data_root": str(data_root)}
    test_subset = {"ann_file": str(ann_file_root / "test.json"), "data_root": str(data_root)}

    val_subset = {"ann_file": str(ann_file_root / "val.json"), "data_root": str(data_root)}
    return train_subset, test_subset, val_subset
