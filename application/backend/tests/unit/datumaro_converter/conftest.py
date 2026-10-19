# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from datumaro.experimental import Dataset

from app.datumaro_converter import SampleMode, convert_dataset
from app.models import (
    DatasetItem,
    DatasetItemAnnotation,
    DatasetItemSubset,
    FullImage,
    Image,
    Label,
    LabelReference,
    MediaType,
    Point,
    Polygon,
    Rectangle,
    Shape,
    Task,
    TaskType,
)
from app.models.media import ImageFormat


@pytest.fixture
def fxt_project_labels():
    project_id = uuid4()
    return [
        Label(id=uuid4(), project_id=project_id, name="cat", color="#00FF00", hotkey="c"),
        Label(id=uuid4(), project_id=project_id, name="dog", color="#FF0000", hotkey="d"),
    ]


@pytest.fixture
def fxt_dataset_item():
    def _create_dataset_item(
        project_id: UUID,
        annotation_data: list[DatasetItemAnnotation] | None = None,
    ) -> DatasetItem:
        return DatasetItem(
            id=uuid4(),
            project_id=project_id,
            annotation_data=annotation_data or [],
            subset=DatasetItemSubset.TRAINING,
            subset_assigned_at=None,
            user_reviewed=False,
            prediction_model_id=None,
        )

    return _create_dataset_item


@pytest.fixture
def fxt_dataset_item_annotation():
    def _create_dataset_item_annotation(labels: list[str], shape: Shape) -> DatasetItemAnnotation:
        return DatasetItemAnnotation(labels=[LabelReference(id=UUID(label)) for label in labels], shape=shape)

    return _create_dataset_item_annotation


@pytest.fixture
def fxt_detection_dataset_item(fxt_dataset_item, fxt_dataset_item_annotation):
    def _create_detection_dataset_item(
        project_id: UUID, label: str, x: int, y: int, width: int, height: int
    ) -> DatasetItem:
        return fxt_dataset_item(
            project_id=project_id,
            annotation_data=[
                fxt_dataset_item_annotation(labels=[label], shape=Rectangle(x=x, y=y, width=width, height=height))
            ],
        )

    return _create_detection_dataset_item


@pytest.fixture
def fxt_classification_dataset_item(fxt_dataset_item, fxt_dataset_item_annotation):
    def _create_classification_dataset_item(project_id: UUID, label: str) -> DatasetItem:
        return fxt_dataset_item(
            project_id=project_id,
            annotation_data=[fxt_dataset_item_annotation(labels=[label], shape=FullImage())],
        )

    return _create_classification_dataset_item


@pytest.fixture
def fxt_multilabel_classification_dataset_item(fxt_dataset_item, fxt_dataset_item_annotation):
    def _create_classification_dataset_item(project_id: UUID, labels: list[str]) -> DatasetItem:
        return fxt_dataset_item(
            project_id=project_id,
            annotation_data=[fxt_dataset_item_annotation(labels=labels, shape=FullImage())],
        )

    return _create_classification_dataset_item


@pytest.fixture
def fxt_instance_segmentation_dataset_item(fxt_dataset_item, fxt_dataset_item_annotation):
    def _create_instance_segmentation_dataset_item(
        project_id: UUID, annotations: list[tuple[str, list[list[int]]]]
    ) -> DatasetItem:
        return fxt_dataset_item(
            project_id=project_id,
            annotation_data=[
                fxt_dataset_item_annotation(
                    labels=[label], shape=Polygon(points=[Point(x=p[0], y=p[1]) for p in coords])
                )
                for (label, coords) in annotations
            ],
        )

    return _create_instance_segmentation_dataset_item


_TASKS: dict[str, Task] = {
    "multiclass": Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
    "multilabel": Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=False),
    "detection": Task(task_type=TaskType.DETECTION),
    "instance_segmentation": Task(task_type=TaskType.INSTANCE_SEGMENTATION),
}


@pytest.fixture(params=list(_TASKS))
def fxt_task_name(request: pytest.FixtureRequest) -> str:
    """Name of a task type supported by the converter."""
    return request.param


@pytest.fixture
def fxt_task_dataset(
    fxt_classification_dataset_item,
    fxt_multilabel_classification_dataset_item,
    fxt_detection_dataset_item,
    fxt_instance_segmentation_dataset_item,
) -> Callable[..., tuple[Dataset, list[list[int]]]]:
    """
    Build a converted dataset for a task whose annotations use the given label indices.
    Returns a callable producing the dataset and the expected label indices per sample.
    """

    def _build(
        task_name: str,
        label_indices: list[int],
        sample_mode: SampleMode,
        num_labels: int = 300,
        media_paths: list[str] | None = None,
    ) -> tuple[Dataset, list[list[int]]]:
        project_id = uuid4()
        labels = [
            Label(id=uuid4(), project_id=project_id, name=f"label_{i}", color="#00FF00") for i in range(num_labels)
        ]
        label_ids = [str(labels[idx].id) for idx in label_indices]
        polygon = [[0, 0], [10, 0], [10, 10]]
        match task_name:
            case "multiclass":
                items = [fxt_classification_dataset_item(project_id, label_id) for label_id in label_ids]
                expected = [[idx] for idx in label_indices]
            case "multilabel":
                items = [fxt_multilabel_classification_dataset_item(project_id, label_ids)]
                expected = [label_indices]
            case "detection":
                items = [fxt_detection_dataset_item(project_id, label_id, 4, 5, 10, 10) for label_id in label_ids]
                expected = [[idx] for idx in label_indices]
            case "instance_segmentation":
                items = [
                    fxt_instance_segmentation_dataset_item(project_id, [(label_id, polygon)]) for label_id in label_ids
                ]
                expected = [[idx] for idx in label_indices]
            case _:
                raise ValueError(f"Unknown task: {task_name}")
        media = Image(
            id=uuid4(),
            project_id=project_id,
            type=MediaType.IMAGE,
            name="test",
            width=200,
            height=100,
            format=ImageFormat.JPG,
            size=1024,
            source_id=None,
        )
        paths = media_paths or [f"path{i}" for i in range(len(items))]
        dataset = convert_dataset(
            task=_TASKS[task_name],
            labels=labels,
            get_dataset_items_and_media=MagicMock(side_effect=[[(item, media) for item in items], []]),
            get_media_path=MagicMock(side_effect=paths[: len(items)]),
            sample_mode=sample_mode,
        )
        return dataset, expected

    return _build
