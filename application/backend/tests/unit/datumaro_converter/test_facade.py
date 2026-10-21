# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, call
from uuid import uuid4

import numpy as np
import polars as pl
import pytest
from datumaro.experimental import LazyImage

from app.datumaro_converter import (
    DetectionImportExportSample,
    DetectionTrainingSample,
    InstanceSegmentationImportExportSample,
    InstanceSegmentationTrainingSample,
    MulticlassClassificationImportExportSample,
    MulticlassClassificationTrainingSample,
    MultilabelClassificationImportExportSample,
    MultilabelClassificationTrainingSample,
    SampleMode,
    convert_dataset,
)
from app.models import DatasetItemSubset, Image, MediaType, Task, TaskType
from app.models.media import ImageFormat


@pytest.mark.parametrize(
    "sample_mode, sample_type, media_attr",
    [
        (SampleMode.TRAINING, DetectionTrainingSample, "image"),
        (SampleMode.IMPORT_EXPORT, DetectionImportExportSample, "media"),
    ],
)
def test_convert_detection_dataset(
    sample_mode, sample_type, media_attr, fxt_project_labels, fxt_detection_dataset_item
) -> None:
    project_id = uuid4()
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
    dataset_item_1 = fxt_detection_dataset_item(project_id, str(fxt_project_labels[0].id), 4, 5, 10, 10)
    dataset_item_2 = fxt_detection_dataset_item(project_id, str(fxt_project_labels[1].id), 14, 35, 10, 10)
    get_dataset_items_and_media = MagicMock(side_effect=[[(dataset_item_1, media), (dataset_item_2, media)], []])
    get_media_path = MagicMock(side_effect=["path1", "path2"])

    dataset = convert_dataset(
        task=Task(task_type=TaskType.DETECTION),
        labels=fxt_project_labels,
        get_dataset_items_and_media=get_dataset_items_and_media,
        get_media_path=get_media_path,
        sample_mode=sample_mode,
    )

    media_attr_info = f"{media_attr}_info"
    assert len(dataset) == 2
    assert (
        isinstance(dataset[0], sample_type)
        and getattr(dataset[0], media_attr) == LazyImage("path1")
        and np.array_equal(dataset[0].label, np.array([0]))
        and getattr(dataset[0], media_attr_info).width == 200
        and getattr(dataset[0], media_attr_info).height == 100
    )
    assert (
        isinstance(dataset[1], sample_type)
        and getattr(dataset[1], media_attr) == LazyImage("path2")
        and np.array_equal(dataset[1].label, np.array([1]))
        and getattr(dataset[1], media_attr_info).width == 200
        and getattr(dataset[1], media_attr_info).height == 100
    )
    assert get_media_path.call_count == 2
    get_media_path.assert_has_calls([call(media)] * 2)


@pytest.mark.parametrize(
    "sample_mode, sample_type, media_attr",
    [
        (SampleMode.TRAINING, MulticlassClassificationTrainingSample, "image"),
        (SampleMode.IMPORT_EXPORT, MulticlassClassificationImportExportSample, "media"),
    ],
)
def test_convert_multiclass_classification_dataset(
    sample_mode, sample_type, media_attr, fxt_project_labels, fxt_classification_dataset_item
) -> None:
    project_id = uuid4()
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
    dataset_item_1 = fxt_classification_dataset_item(project_id, str(fxt_project_labels[0].id))
    dataset_item_2 = fxt_classification_dataset_item(project_id, str(fxt_project_labels[1].id))
    get_dataset_items_and_media = MagicMock(side_effect=[[(dataset_item_1, media), (dataset_item_2, media)], []])
    get_media_path = MagicMock(side_effect=["path1", "path2"])

    dataset = convert_dataset(
        task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
        labels=fxt_project_labels,
        get_dataset_items_and_media=get_dataset_items_and_media,
        get_media_path=get_media_path,
        sample_mode=sample_mode,
    )

    media_attr_info = f"{media_attr}_info"
    assert len(dataset) == 2
    assert (
        isinstance(dataset[0], sample_type)
        and getattr(dataset[0], media_attr) == LazyImage("path1")
        and dataset[0].label == 0
        and getattr(dataset[0], media_attr_info).width == 200
        and getattr(dataset[0], media_attr_info).height == 100
    )
    assert (
        isinstance(dataset[1], sample_type)
        and getattr(dataset[1], media_attr) == LazyImage("path2")
        and dataset[1].label == 1
        and getattr(dataset[1], media_attr_info).width == 200
        and getattr(dataset[1], media_attr_info).height == 100
    )
    assert get_media_path.call_count == 2
    get_media_path.assert_has_calls([call(media)] * 2)


@pytest.mark.parametrize("sample_mode", [SampleMode.TRAINING, SampleMode.IMPORT_EXPORT])
def test_convert_dataset_label_index_above_uint8(fxt_task_name, sample_mode, fxt_task_dataset) -> None:
    """Class indices above 255 must survive conversion (projects with more than 256 labels)."""
    dataset, expected_labels = fxt_task_dataset(fxt_task_name, label_indices=[255, 256, 299], sample_mode=sample_mode)
    label_dtype = dataset.df["label"].dtype
    assert (label_dtype.inner if isinstance(label_dtype, pl.List) else label_dtype) == pl.UInt16()
    assert [np.atleast_1d(dataset[i].label).tolist() for i in range(len(dataset))] == expected_labels


@pytest.mark.parametrize(
    "sample_mode, sample_type, media_attr",
    [
        (SampleMode.TRAINING, MultilabelClassificationTrainingSample, "image"),
        (SampleMode.IMPORT_EXPORT, MultilabelClassificationImportExportSample, "media"),
    ],
)
def test_convert_multilabel_classification_dataset_item(
    sample_mode, sample_type, media_attr, fxt_project_labels, fxt_multilabel_classification_dataset_item
) -> None:
    project_id = uuid4()
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
    dataset_item = fxt_multilabel_classification_dataset_item(
        project_id, [str(fxt_project_labels[0].id), str(fxt_project_labels[1].id)]
    )
    dataset_item_empty_label_training = fxt_multilabel_classification_dataset_item(project_id, [])
    dataset_item_empty_label_training.subset = DatasetItemSubset.TRAINING
    dataset_item_empty_label_validation = fxt_multilabel_classification_dataset_item(project_id, [])
    dataset_item_empty_label_validation.subset = DatasetItemSubset.VALIDATION
    get_dataset_items_and_media = MagicMock(
        side_effect=[
            [
                (dataset_item, media),
                (dataset_item_empty_label_training, media),
                (dataset_item_empty_label_validation, media),
            ],
            [],
        ]
    )
    get_media_path = MagicMock(side_effect=["path1", "path2", "path3"])

    dataset = convert_dataset(
        task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=False),
        labels=fxt_project_labels,
        get_dataset_items_and_media=get_dataset_items_and_media,
        get_media_path=get_media_path,
        sample_mode=sample_mode,
    )

    media_attr_info = f"{media_attr}_info"
    assert len(dataset) == 2
    assert (
        isinstance(dataset[0], sample_type)
        and getattr(dataset[0], media_attr) == LazyImage("path1")
        and np.array_equal(dataset[0].label, np.array([0, 1]))
        and getattr(dataset[0], media_attr_info).width == 200
        and getattr(dataset[0], media_attr_info).height == 100
    )
    assert isinstance(dataset[1], sample_type) and getattr(dataset[1], media_attr) == LazyImage("path3")
    get_media_path.assert_has_calls([call(media)] * 3)


@pytest.mark.parametrize(
    "sample_mode, sample_type, media_attr",
    [
        (SampleMode.TRAINING, InstanceSegmentationTrainingSample, "image"),
        (SampleMode.IMPORT_EXPORT, InstanceSegmentationImportExportSample, "media"),
    ],
)
def test_convert_instance_segmentation_dataset(
    sample_mode, sample_type, media_attr, fxt_project_labels, fxt_instance_segmentation_dataset_item
) -> None:
    project_id = uuid4()
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
    dataset_item_1 = fxt_instance_segmentation_dataset_item(
        project_id,
        [
            (str(fxt_project_labels[0].id), [[0, 0], [10, 0], [10, 10], [0, 10]]),
            (str(fxt_project_labels[0].id), [[20, 20], [30, 20], [30, 30]]),
        ],
    )
    dataset_item_2 = fxt_instance_segmentation_dataset_item(
        project_id,
        [
            (str(fxt_project_labels[1].id), [[4, 6], [14, 6], [14, 16], [4, 16]]),
            (str(fxt_project_labels[1].id), [[49, 20], [59, 20], [49, 30], [59, 30]]),
        ],
    )
    get_dataset_items_and_media = MagicMock(side_effect=[[(dataset_item_1, media), (dataset_item_2, media)], []])
    get_media_path = MagicMock(side_effect=["path1", "path2"])

    dataset = convert_dataset(
        task=Task(task_type=TaskType.INSTANCE_SEGMENTATION),
        labels=fxt_project_labels,
        get_dataset_items_and_media=get_dataset_items_and_media,
        get_media_path=get_media_path,
        sample_mode=sample_mode,
    )

    media_attr_info = f"{media_attr}_info"
    assert len(dataset) == 2
    assert (
        isinstance(dataset[0], sample_type)
        and getattr(dataset[0], media_attr) == LazyImage("path1")
        and np.array_equal(dataset[0].label, np.array([0, 0]))
        and getattr(dataset[0], media_attr_info).width == 200
        and getattr(dataset[0], media_attr_info).height == 100
    )
    assert (
        isinstance(dataset[1], sample_type)
        and getattr(dataset[1], media_attr) == LazyImage("path2")
        and np.array_equal(dataset[1].label, np.array([1, 1]))
        and getattr(dataset[1], media_attr_info).width == 200
        and getattr(dataset[1], media_attr_info).height == 100
    )
    assert get_media_path.call_count == 2
    get_media_path.assert_has_calls([call(media)] * 2)
