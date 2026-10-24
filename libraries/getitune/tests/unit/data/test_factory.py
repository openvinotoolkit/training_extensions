# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test Factory classes for dataset and transforms."""

import pytest
from datumaro.experimental import Dataset

from getitune.config.data import IntensityConfig, SubsetConfig
from getitune.data.dataset.classification import (
    MulticlassClsDataset,
    MultilabelClsDataset,
)
from getitune.data.dataset.detection import DetectionDataset
from getitune.data.dataset.instance_segmentation import InstanceSegDataset
from getitune.data.dataset.segmentation import SegmentationDataset
from getitune.data.factory import DatasetFactory, TransformLibFactory
from getitune.types.task import TaskType


class TestDatasetFactory:
    @pytest.mark.parametrize(
        ("task_type", "dataset_cls", "dm_subset_fxt_name"),
        [
            (TaskType.MULTI_CLASS_CLS, MulticlassClsDataset, "fxt_mock_classification_dm_subset"),
            (TaskType.MULTI_LABEL_CLS, MultilabelClsDataset, "fxt_mock_classification_dm_subset"),
            (TaskType.DETECTION, DetectionDataset, "fxt_mock_detection_dm_subset"),
            (TaskType.INSTANCE_SEGMENTATION, InstanceSegDataset, "fxt_mock_segmentation_dm_subset"),
            (TaskType.SEMANTIC_SEGMENTATION, SegmentationDataset, "fxt_mock_segmentation_dm_subset"),
        ],
    )
    def test_create(
        self,
        request,
        task_type,
        dataset_cls,
        dm_subset_fxt_name,
        mocker,
    ) -> None:
        mocker.patch.object(TransformLibFactory, "generate", return_value=None)
        dm_subset = request.getfixturevalue(dm_subset_fxt_name)
        mock_schema = mocker.MagicMock()
        mock_label = mocker.MagicMock()
        mock_label.categories.labels = []
        mock_schema.attributes = {"label": mock_label, "masks": mock_label}
        dm_subset.schema = mock_schema
        cfg_subset = mocker.MagicMock(spec=SubsetConfig)
        cfg_subset.intensity = IntensityConfig()
        mocker.patch.object(Dataset, "convert_to_schema", return_value=dm_subset)

        assert isinstance(
            DatasetFactory.create(
                task=task_type,
                dm_subset=dm_subset,
                cfg_subset=cfg_subset,
            ),
            dataset_cls,
        )
