# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for classification_new dataset."""

from __future__ import annotations

from unittest.mock import Mock

import polars as pl
import pytest
import torch
from datumaro.experimental import Dataset, register_sample
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.dataset import Sample
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset, image_field, image_info_field, label_field, subset_field
from torchvision import tv_tensors

from getitune.data.dataset.base import _default_collate_fn
from getitune.data.dataset.classification import MulticlassClsDataset
from getitune.data.entity.sample import ClassificationSample

_NUM_CLASSES_ABOVE_UINT8 = 300


@register_sample
class _LegacyUInt8ClassificationSample(Sample):
    """Classification sample with the previous ``UInt8`` label storage, used to check schema widening."""

    subset: Subset = subset_field()
    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(pl.UInt8())
    dm_image_info: DmImageInfo = image_info_field()


def _make_dm_dataset(sample_cls: type[Sample], labels: list[int], num_classes: int) -> Dataset:
    categories = {"label": LabelCategories(labels=tuple(f"class_{i}" for i in range(num_classes)))}
    dm_dataset = Dataset(sample_cls, categories=categories)  # type: ignore[arg-type]
    label_dtype = torch.uint16 if sample_cls is ClassificationSample else torch.uint8
    for label in labels:
        dm_dataset.append(
            sample_cls(
                image=tv_tensors.Image(torch.zeros(3, 8, 8, dtype=torch.uint8)),
                label=torch.tensor(label, dtype=label_dtype),
                dm_image_info=DmImageInfo(width=8, height=8),
                subset=Subset.TRAINING,
            ),
        )
    return dm_dataset


class TestMulticlassClsDatasetLabelRange:
    """Test that multi-class labels are not limited to the UInt8 range."""

    def test_label_field_is_uint16(self) -> None:
        assert ClassificationSample.label.dtype == pl.UInt16()  # type: ignore[attr-defined]

    def test_labels_above_255(self) -> None:
        labels = [0, 255, 256, _NUM_CLASSES_ABOVE_UINT8 - 1]
        dm_dataset = _make_dm_dataset(ClassificationSample, labels, _NUM_CLASSES_ABOVE_UINT8)

        dataset = MulticlassClsDataset(dm_subset=dm_dataset, transforms=[])

        assert len(dataset.label_info.label_names) == _NUM_CLASSES_ABOVE_UINT8
        assert [int(dataset[i].label) for i in range(len(labels))] == labels

        batch = _default_collate_fn([dataset[i] for i in range(len(labels))])
        assert batch.labels is not None
        assert all(label.dtype == torch.long for label in batch.labels)
        assert [int(label) for label in batch.labels] == labels

        assert dataset.get_idx_list_per_classes() == {label: [idx] for idx, label in enumerate(labels)}

    @pytest.mark.parametrize("labels", [[0, 1, 2], [0, 128, 255]])
    def test_uint8_stored_labels_are_widened(self, labels: list[int]) -> None:
        dm_dataset = _make_dm_dataset(_LegacyUInt8ClassificationSample, labels, num_classes=256)

        dataset = MulticlassClsDataset(dm_subset=dm_dataset, transforms=[])

        assert [int(dataset[i].label) for i in range(len(labels))] == labels
        assert dataset.get_idx_list_per_classes() == {label: [idx] for idx, label in enumerate(labels)}


class TestMulticlassClsDataset:
    """Test MulticlassClsDataset class."""

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

    def test_get_idx_list_per_classes_single_class(self):
        """Test get_idx_list_per_classes with single class."""
        # Mock dataset items with labels
        mock_items = []
        for _ in range(5):
            mock_item = Mock()
            mock_item.label.item.return_value = 0  # All items have label 0
            mock_items.append(mock_item)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = MulticlassClsDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        # Override length for this test
        dataset.dm_subset.__len__ = Mock(return_value=5)

        result = dataset.get_idx_list_per_classes()

        expected = {0: [0, 1, 2, 3, 4]}
        assert result == expected

    def test_get_idx_list_per_classes_string_labels(self):
        """Test get_idx_list_per_classes with use_string_label=True."""
        # Create three items with labels 0, 1, 2
        mock_items = []
        for lbl in [0, 1, 2, 1, 0]:
            mock_item = Mock()
            mock_item.label.item.return_value = lbl
            mock_items.append(mock_item)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = MulticlassClsDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        # Override length for this test
        dataset.dm_subset.__len__ = Mock(return_value=5)

        result = dataset.get_idx_list_per_classes(use_string_label=True)

        expected = {
            "class_0": [0, 4],
            "class_1": [1, 3],
            "class_2": [2],
        }
        assert result == expected
