# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for base_new VisionDataset."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch
from datumaro.experimental import Dataset

from getitune import LabelInfo
from getitune.data.dataset.base import VisionDataset, _default_collate_fn
from getitune.data.entity.sample import BaseSample, SampleBatch


class TestDefaultCollateFn:
    """Test _default_collate_fn function."""

    def test_collate_with_torch_tensors(self):
        """Test collating items with torch tensor images."""
        # Create mock samples with torch tensor images
        sample1 = Mock(spec=BaseSample)
        sample1.image = torch.randn(3, 224, 224)
        sample1.label = torch.tensor(0)
        sample1.masks = None
        sample1.bboxes = None
        sample1.keypoints = None
        sample1.img_info = None

        sample2 = Mock(spec=BaseSample)
        sample2.image = torch.randn(3, 224, 224)
        sample2.label = torch.tensor(1)
        sample2.masks = None
        sample2.bboxes = None
        sample2.keypoints = None
        sample2.img_info = None

        items = [sample1, sample2]
        result = _default_collate_fn(items)

        assert isinstance(result, SampleBatch)
        assert result.batch_size == 2
        assert isinstance(result.images, torch.Tensor)
        assert result.images.shape == (2, 3, 224, 224)
        assert result.images.dtype == torch.float32
        assert result.labels == [torch.tensor(0), torch.tensor(1)]

    def test_collate_with_different_image_shapes(self):
        """Test collating items with different image shapes raises RuntimeError."""
        sample1 = Mock(spec=BaseSample)
        sample1.image = torch.randn(3, 224, 224)
        sample1.label = None
        sample1.masks = None
        sample1.bboxes = None
        sample1.keypoints = None
        sample1.img_info = None

        sample2 = Mock(spec=BaseSample)
        sample2.image = torch.randn(3, 256, 256)
        sample2.label = None
        sample2.masks = None
        sample2.bboxes = None
        sample2.keypoints = None
        sample2.img_info = None

        items = [sample1, sample2]
        # torch.stack requires same-size tensors; different shapes mean
        # the resize/augmentation pipeline is misconfigured.
        with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
            _default_collate_fn(items)

    def test_collate_rejects_unprocessed_16bit_images(self):
        """Test that int32 tensors (simulating unprocessed 16-bit images) are rejected."""
        sample = Mock(spec=BaseSample)
        sample.image = torch.randint(0, 65536, (3, 32, 32), dtype=torch.int32)
        sample.label = torch.tensor(0)
        sample.masks = None
        sample.bboxes = None
        sample.keypoints = None
        sample.img_info = None

        with pytest.raises(TypeError, match="high-bit-depth image"):
            _default_collate_fn([sample])

    def test_collate_rejects_int16_images(self):
        """Test that int16 tensors (unprocessed signed 16-bit) are rejected."""
        sample = Mock(spec=BaseSample)
        sample.image = torch.randint(-1000, 1000, (3, 32, 32), dtype=torch.int16)
        sample.label = torch.tensor(0)
        sample.masks = None
        sample.bboxes = None
        sample.keypoints = None
        sample.img_info = None

        with pytest.raises(TypeError, match="high-bit-depth image"):
            _default_collate_fn([sample])

    @pytest.mark.parametrize("label_dtype", [torch.uint8, torch.uint16])
    def test_collate_casts_integer_labels_to_long(self, label_dtype: torch.dtype):
        """Test that uint8/uint16 labels are cast to long during collation."""
        sample1 = Mock(spec=BaseSample)
        sample1.image = torch.randn(3, 224, 224)
        sample1.label = torch.tensor([0, 1], dtype=label_dtype)
        sample1.masks = None
        sample1.bboxes = None
        sample1.keypoints = None
        sample1.img_info = None

        sample2 = Mock(spec=BaseSample)
        sample2.image = torch.randn(3, 224, 224)
        sample2.label = torch.tensor([2], dtype=label_dtype)
        sample2.masks = None
        sample2.bboxes = None
        sample2.keypoints = None
        sample2.img_info = None

        result = _default_collate_fn([sample1, sample2])

        assert isinstance(result, SampleBatch)
        assert result.labels is not None
        for label in result.labels:
            assert label.dtype == torch.long, f"Expected torch.long but got {label.dtype}"
        # Verify values are preserved after casting
        assert torch.equal(result.labels[0], torch.tensor([0, 1], dtype=torch.long))
        assert torch.equal(result.labels[1], torch.tensor([2], dtype=torch.long))

    @pytest.mark.parametrize("label_dtype", [torch.uint8, torch.uint16])
    def test_collate_casts_empty_integer_labels_to_long(self, label_dtype: torch.dtype):
        """Test that empty uint8/uint16 labels (unannotated images) are cast to long."""
        sample1 = Mock(spec=BaseSample)
        sample1.image = torch.randn(3, 64, 64)
        sample1.label = torch.tensor([1], dtype=label_dtype)
        sample1.masks = None
        sample1.bboxes = None
        sample1.keypoints = None
        sample1.img_info = None

        sample2 = Mock(spec=BaseSample)
        sample2.image = torch.randn(3, 64, 64)
        sample2.label = torch.zeros(0, dtype=label_dtype)  # empty label (no annotations)
        sample2.masks = None
        sample2.bboxes = None
        sample2.keypoints = None
        sample2.img_info = None

        result = _default_collate_fn([sample1, sample2])

        assert result.labels is not None
        for label in result.labels:
            assert label.dtype == torch.long
        assert result.labels[1].shape == (0,)


class TestVisionDataset:
    """Test VisionDataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dm_subset = Mock(spec=Dataset)
        self.mock_dm_subset.__len__ = Mock(return_value=100)

        # Mock schema attributes for label_info
        mock_schema = Mock()
        mock_attributes = {"label": Mock()}
        mock_attributes["label"].categories = Mock()
        # Configure labels to be a list with proper length support
        mock_attributes["label"].categories.labels = ["class_0", "class_1", "class_2"]
        mock_schema.attributes = mock_attributes
        self.mock_dm_subset.schema = mock_schema

        self.mock_transforms = Mock()

    def test_apply_transforms_with_compose(self):
        """Test _apply_transforms with Compose transforms."""
        from torchvision.transforms.v2 import Compose

        mock_compose = Mock(spec=Compose)
        mock_entity = Mock(spec=BaseSample)
        mock_entity.image = torch.rand(3, 32, 32, dtype=torch.float32)
        mock_result = Mock()
        mock_compose.return_value = mock_result

        dataset = VisionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=mock_compose,
        )

        result = dataset._apply_transforms(mock_entity)

        mock_compose.assert_called_once_with(mock_entity)
        assert result == mock_result

    def test_apply_transforms_with_callable(self):
        """Test _apply_transforms with callable transform."""
        mock_transform = Mock()
        mock_entity = Mock(spec=BaseSample)
        mock_entity.image = torch.rand(3, 32, 32, dtype=torch.float32)
        mock_result = Mock()
        mock_transform.return_value = mock_result

        dataset = VisionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=mock_transform,
        )

        result = dataset._apply_transforms(mock_entity)

        mock_transform.assert_called_once_with(mock_entity)
        assert result == mock_result

    def test_apply_transforms_with_list(self):
        """Test _apply_transforms with list of transforms."""
        transform1 = Mock()
        transform2 = Mock()

        mock_entity = Mock(spec=BaseSample)
        mock_entity.image = torch.rand(3, 32, 32, dtype=torch.float32)
        intermediate_result = Mock()
        final_result = Mock()

        transform1.return_value = intermediate_result
        transform2.return_value = final_result

        dataset = VisionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=[transform1, transform2],
        )

        result = dataset._apply_transforms(mock_entity)

        transform1.assert_called_once_with(mock_entity)
        transform2.assert_called_once_with(intermediate_result)
        assert result == final_result

    def test_apply_transforms_with_list_returns_none(self):
        """Test _apply_transforms with list that returns None."""
        transform1 = Mock()
        transform2 = Mock()

        mock_entity = Mock(spec=BaseSample)
        mock_entity.image = torch.rand(3, 32, 32, dtype=torch.float32)
        transform1.return_value = None  # First transform returns None

        dataset = VisionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=[transform1, transform2],
        )

        result = dataset._apply_transforms(mock_entity)

        transform1.assert_called_once_with(mock_entity)
        transform2.assert_not_called()  # Should not be called since first returned None
        assert result is None

    def test_iterable_transforms_with_non_list(self):
        """Test _iterable_transforms with non-list iterable raises TypeError."""
        dataset = VisionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        mock_entity = Mock(spec=BaseSample)
        mock_entity.image = torch.rand(3, 32, 32, dtype=torch.float32)
        dataset.transforms = "not_a_list"  # String is iterable but not a list

        with pytest.raises(TypeError):
            dataset._iterable_transforms(mock_entity)

    def test_getitem_success(self):
        """Test __getitem__ with successful retrieval."""
        mock_item = Mock()
        self.mock_dm_subset.__getitem__ = Mock(return_value=mock_item)

        mock_transformed_item = Mock(spec=BaseSample)
        mock_transformed_item.image = torch.rand(3, 32, 32)

        dataset = VisionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        with patch.object(dataset, "_apply_transforms", return_value=mock_transformed_item):
            result = dataset[5]

            self.mock_dm_subset.__getitem__.assert_called_once_with(5)
            assert result == mock_transformed_item

    def test_getitem_with_refetch(self):
        """Test __getitem__ with failed first attempt requiring refetch."""
        mock_item = Mock()
        self.mock_dm_subset.__getitem__ = Mock(return_value=mock_item)

        dataset = VisionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            max_refetch=2,
        )

        mock_transformed_item = Mock(spec=BaseSample)

        # First call returns None, second returns valid item
        with patch.object(dataset, "_apply_transforms", side_effect=[None, mock_transformed_item]):
            result = dataset[5]

            assert result == mock_transformed_item
            assert dataset._apply_transforms.call_count == 2

    def test_collate_fn_property(self):
        """Test collate_fn property returns _default_collate_fn."""
        dataset = VisionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        collate = dataset.collate_fn
        assert collate is _default_collate_fn

    def test_repr_default(self):
        """Test repr on a default dataset reports class name and zero classes."""
        dataset = VisionDataset(dm_subset=self.mock_dm_subset)

        assert "VisionDataset" in repr(dataset)
        assert "num_samples=100" in repr(dataset)
        assert "num_classes=0" in repr(dataset)

    def test_repr_with_label_info(self):
        """Test repr reflects num_classes after label_info is populated."""
        dataset = VisionDataset(dm_subset=self.mock_dm_subset)
        dataset.label_info = LabelInfo(
            label_names=["cat", "dog"],
            label_groups=[["cat", "dog"]],
            label_ids=["0", "1"],
        )

        assert "num_classes=2" in repr(dataset)

    def test_repr_uses_subclass_name(self):
        """Test repr uses the subclass name, not the base class name."""

        class FakeDataset(VisionDataset):
            pass

        dataset = FakeDataset(dm_subset=self.mock_dm_subset)

        assert "FakeDataset" in repr(dataset)

    def test_describe_basic(self):
        """Test describe returns expected keys and values on a default dataset."""
        dataset = VisionDataset(dm_subset=self.mock_dm_subset)

        result = dataset.describe()

        assert isinstance(result, dict)
        assert result["class_name"] == "VisionDataset"
        assert result["num_samples"] == 100
        assert result["num_classes"] == 0
        assert result["label_names"] == []
        assert result["has_transforms"] is False

    def test_describe_with_transforms(self):
        """Test describe reports has_transforms=True when transforms are provided."""
        dataset = VisionDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        result = dataset.describe()

        assert result["has_transforms"] is True
