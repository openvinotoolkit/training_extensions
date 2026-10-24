# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for detection dataset."""

from __future__ import annotations

from unittest.mock import Mock

import polars as pl
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from torchvision import tv_tensors

from getitune.data.dataset.base import _default_collate_fn
from getitune.data.dataset.detection import DetectionDataset
from getitune.data.entity.sample import DetectionSample


class TestDetectionDataset:
    """Test DetectionDataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dm_subset = Mock(spec=Dataset)
        self.mock_dm_subset.__len__ = Mock(return_value=10)
        self.mock_dm_subset.convert_to_schema = Mock(return_value=self.mock_dm_subset)

        # Mock schema attributes for label_info
        mock_schema = Mock()
        mock_attributes = {"label": Mock()}
        mock_attributes["label"].categories = Mock()
        # Configure labels to be a list with proper length support
        mock_attributes["label"].categories.labels = ["class_0", "class_1", "class_2"]
        mock_schema.attributes = mock_attributes
        self.mock_dm_subset.schema = mock_schema

        # Mock label_categories used by the dataset implementation
        mock_label_categories = Mock()
        mock_label_categories.labels = ["class_0", "class_1", "class_2"]
        self.mock_dm_subset.label_categories = mock_label_categories

        self.mock_transforms = Mock()

    def test_get_idx_list_per_classes_multiple_classes_per_item(self):
        """Test get_idx_list_per_classes with multiple classes per item."""
        # Mock dataset items with multiple labels per item
        mock_items = []
        # Item 0: classes [0, 1]
        mock_item0 = Mock()
        mock_item0.label.tolist.return_value = [0, 1]
        mock_items.append(mock_item0)

        # Item 1: class [1]
        mock_item1 = Mock()
        mock_item1.label.tolist.return_value = [1]
        mock_items.append(mock_item1)

        # Item 2: classes [0, 2]
        mock_item2 = Mock()
        mock_item2.label.tolist.return_value = [0, 2]
        mock_items.append(mock_item2)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = DetectionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        # Override length for this test
        dataset.dm_subset.__len__ = Mock(return_value=3)

        result = dataset.get_idx_list_per_classes()

        expected = {
            0: [0, 2],  # Items 0 and 2 contain class 0
            1: [0, 1],  # Items 0 and 1 contain class 1
            2: [2],  # Item 2 contains class 2
        }
        assert result == expected

    def test_get_idx_list_per_classes_string_labels(self):
        """Test get_idx_list_per_classes with use_string_label=True."""
        # Mock dataset items with multiple labels per item
        mock_items = []
        # Item 0: classes [0, 1]
        mock_item0 = Mock()
        mock_item0.label.tolist.return_value = [0, 1]
        mock_items.append(mock_item0)

        # Item 1: class [1]
        mock_item1 = Mock()
        mock_item1.label.tolist.return_value = [1]
        mock_items.append(mock_item1)

        # Item 2: classes [0, 2]
        mock_item2 = Mock()
        mock_item2.label.tolist.return_value = [0, 2]
        mock_items.append(mock_item2)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = DetectionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        # Override length for this test
        dataset.dm_subset.__len__ = Mock(return_value=3)

        result = dataset.get_idx_list_per_classes(use_string_label=True)

        expected = {
            "class_0": [0, 2],
            "class_1": [0, 1],
            "class_2": [2],
        }
        assert result == expected


class TestDetectionDatasetLabelRange:
    """Test that detection labels are not limited to the UInt8 range."""

    def test_label_field_is_uint16(self) -> None:
        assert DetectionSample.label.dtype == pl.UInt16()  # type: ignore[attr-defined]

    def test_labels_above_255(self) -> None:
        num_classes = 300
        labels = [[0, 255], [256, num_classes - 1], []]
        categories = {"label": LabelCategories(labels=tuple(f"class_{i}" for i in range(num_classes)))}
        dm_dataset = Dataset(DetectionSample, categories=categories)  # type: ignore[arg-type]
        for sample_labels in labels:
            dm_dataset.append(
                DetectionSample(
                    image=tv_tensors.Image(torch.zeros(3, 8, 8, dtype=torch.uint8)),
                    label=torch.tensor(sample_labels, dtype=torch.uint16),
                    bboxes=torch.tensor([[0.0, 0.0, 4.0, 4.0]] * len(sample_labels)).reshape(-1, 4),
                    dm_image_info=DmImageInfo(width=8, height=8),
                    subset=Subset.TRAINING,
                ),
            )
        dataset = DetectionDataset(dm_subset=dm_dataset, transforms=[])
        batch = _default_collate_fn([dataset[i] for i in range(len(labels))])
        assert batch.labels is not None
        assert all(label.dtype == torch.long for label in batch.labels)
        assert [label.tolist() for label in batch.labels] == labels
        assert dataset.get_idx_list_per_classes() == {0: [0], 255: [0], 256: [1], num_classes - 1: [1]}
